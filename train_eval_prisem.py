"""
Unified EM training + allocation inference script.

This file extends the original Ditto train_ditto.py with two additional EM methods:
  1) Traditional RF baseline: a lightweight Magellan-style RandomForest model over
     parsed DITTO records and string-similarity features.
  2) LLM baseline: Jellyfish / HF causal LM zero-shot inference.

It also implements allocation-based inference. For now, only uniform allocation is
implemented: split the requested number of test samples evenly across methods and
send any remainder to the cheapest method.

Expected DITTO file format:
  <left record>\t<right record>\t<label>
where each record usually looks like:
  COL attr1 VAL value1 COL attr2 VAL value2 ...

Example usage:
  # Original Ditto training behavior only
  python train_ditto.py --mode train_ditto --task Structured/Beer

  # Train Ditto + RF, then run uniform allocation over 1000 test pairs
  python train_ditto.py --mode all --task Structured/Beer --budget 1000 \
      --methods rf,ditto,jellyfish --costs rf:1,ditto:2,jellyfish:3 \
      --jellyfish_backend hf --hf_model NECOUDBFM/Jellyfish-13B \
      --hf_4bit --output allocation_results.jsonl

Notes:
  - Ditto training uses the original ditto_light.ditto.train(...) call.
  - Ditto inference APIs differ across Ditto checkouts. This script tries several
    common helper APIs. If your local Ditto checkout exposes a different predict
    function, edit `predict_ditto_pairs` only.
  - The RF model is intentionally self-contained so it can run directly on DITTO
    train/test files without requiring Abt-Buy tableA/tableB CSVs.
"""

import argparse
import json
import os
import subprocess
import tempfile
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ----------------------------- Optional imports -----------------------------
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.pipeline import Pipeline
except Exception:  # pragma: no cover
    RandomForestClassifier = None
    DictVectorizer = None
    Pipeline = None
    accuracy_score = f1_score = precision_score = recall_score = None

# Ditto imports are kept exactly where the original script expected them.
sys.path.insert(0, "Snippext_public")
try:
    from torch.utils import data as torch_data
    from ditto_light.dataset import DittoDataset
    from ditto_light.summarize import Summarizer
    from ditto_light.knowledge import ProductDKInjector, GeneralDKInjector
    from ditto_light.ditto import train as ditto_train, DittoModel, evaluate as ditto_evaluate
except Exception:  # allows RF/Jellyfish-only runs without Ditto installed
    torch_data = None
    DittoDataset = None
    Summarizer = None
    ProductDKInjector = None
    GeneralDKInjector = None
    ditto_train = None
    DittoModel = None
    ditto_evaluate = None

# HF/OpenAI imports are loaded lazily in the builder functions.

# ----------------------------- DITTO data utils ------------------------------
_LANG_TAG = re.compile(r"@[a-z]{2,3}\b", re.IGNORECASE)
_COL_VAL_RE = re.compile(
    r"COL\s+([^\s]+)\s+VAL\s+(.*?)(?=\s+COL\s+\S+\s+VAL\s+|$)",
    re.IGNORECASE | re.DOTALL,
)
_WS = re.compile(r"\s+")


@dataclass
class PairExample:
    left: str
    right: str
    label: int
    index: int


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_ditto_file(path: str, limit: int = 0) -> List[PairExample]:
    examples: List[PairExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                label = int(parts[-1])
            except ValueError:
                continue
            examples.append(PairExample(parts[0], parts[1], label, i))
            if limit and len(examples) >= limit:
                break
    return examples


def parse_ditto_kv(s: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for m in _COL_VAL_RE.finditer(s):
        attr = m.group(1).strip().replace("_", " ").lower()
        val = m.group(2).strip()
        val = _LANG_TAG.sub("", val).strip()
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        q = re.match(r'^"([^"]*)"', val)
        if q:
            val = q.group(1)
        val = _WS.sub(" ", val).strip()
        if val.lower() in {"none", "nan", "null"}:
            val = ""
        kv[attr] = val
    return kv


def serialize_for_jellyfish(s: str, multiline: bool = False) -> str:
    kv = parse_ditto_kv(s)
    if not kv:
        return s.strip()
    items = [(k, v if v else "N/A") for k, v in kv.items()]
    if multiline:
        return "\n".join(f"- {k}: {v}" for k, v in items)
    return ", ".join(f"{k}: {v}" for k, v in items)


def token_set(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def rf_features(left: str, right: str) -> Dict[str, float]:
    """Magellan-style attribute comparison features for DITTO strings."""
    lkv, rkv = parse_ditto_kv(left), parse_ditto_kv(right)
    attrs = sorted(set(lkv) | set(rkv))
    feats: Dict[str, float] = {}
    feats["record_jaccard"] = jaccard(" ".join(lkv.values()) or left, " ".join(rkv.values()) or right)
    feats["num_attrs_left"] = float(len(lkv))
    feats["num_attrs_right"] = float(len(rkv))
    feats["num_attrs_overlap"] = float(len(set(lkv) & set(rkv)))
    for attr in attrs:
        lv, rv = lkv.get(attr, ""), rkv.get(attr, "")
        feats[f"{attr}__both_present"] = float(bool(lv) and bool(rv))
        feats[f"{attr}__exact"] = float(bool(lv) and bool(rv) and lv.lower() == rv.lower())
        feats[f"{attr}__jaccard"] = jaccard(lv, rv)
        feats[f"{attr}__len_ratio"] = min(len(lv), len(rv)) / max(len(lv), len(rv), 1)
    return feats


# ----------------------------- Costless similarity baseline -------------------
def sim_score_pair(left: str, right: str) -> float:
    """Zero-cost, no-training EM score based on token Jaccard similarity.

    Uses a simple average of:
      1) full-record token Jaccard over all parsed values, and
      2) mean token Jaccard over attributes present on both sides.
    This gives a deterministic fallback matcher for examples not sent to a paid model.
    """
    lkv, rkv = parse_ditto_kv(left), parse_ditto_kv(right)
    left_text = " ".join(lkv.values()) if lkv else left
    right_text = " ".join(rkv.values()) if rkv else right

    record_sim = jaccard(left_text, right_text)

    shared_attrs = sorted(set(lkv) & set(rkv))
    attr_sims = [jaccard(lkv[a], rkv[a]) for a in shared_attrs if lkv.get(a) or rkv.get(a)]
    if attr_sims:
        return 0.5 * record_sim + 0.5 * float(np.mean(attr_sims))
    return record_sim


def predict_sim_pairs(args, examples: List[PairExample]) -> List[int]:
    """Predict with the no-training similarity baseline."""
    threshold = float(args.sim_threshold)
    return [1 if sim_score_pair(ex.left, ex.right) >= threshold else 0 for ex in examples]


# ----------------------------- Metrics/output --------------------------------
def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def write_jsonl(path: str, record: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ----------------------------- RF baseline -----------------------------------
def train_rf_model(train_examples: List[PairExample], seed: int = 0) -> Pipeline:
    if RandomForestClassifier is None or DictVectorizer is None or Pipeline is None:
        raise RuntimeError("scikit-learn is required for --mode train_rf/all.")
    X = [rf_features(ex.left, ex.right) for ex in train_examples]
    y = [ex.label for ex in train_examples]
    model = Pipeline([
        ("vec", DictVectorizer(sparse=False)),
        ("rf", RandomForestClassifier(
            n_estimators=150,
            criterion="gini",
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])
    model.fit(X, y)
    return model


def predict_rf_pairs_with_probs(model: Pipeline, examples: List[PairExample]) -> Tuple[List[int], List[float]]:
    X = [rf_features(ex.left, ex.right) for ex in examples]
    preds = [int(x) for x in model.predict(X)]

    probs_raw = model.predict_proba(X)
    classes = list(model.named_steps["rf"].classes_)
    pos_col = classes.index(1)
    probs = [float(row[pos_col]) for row in probs_raw]
    return preds, probs


def predict_rf_pairs(model: Pipeline, examples: List[PairExample]) -> List[int]:
    preds, _ = predict_rf_pairs_with_probs(model, examples)
    return preds


# ----------------------------- Jellyfish backend -----------------------------
PROMPT_TEMPLATE = (
    "You are an entity matching system.\n"
    "Question: Do the two entity descriptions refer to the same real-world entity?\n"
    "Answer with 'Yes' if they do and 'No' if they do not.\n\n"
    "Entity A:\n{left}\n\nEntity B:\n{right}"
)

JELLYFISH_PROMPT_TEMPLATE = (
    "You are tasked with determining whether two records listed below describe the same entity.\n"
    "Return exactly one token: 'Yes' or 'No'.\n\n"
    "Record A:\n{left}\n\nRecord B:\n{right}\n\nAnswer:"
)


def fmt_safe(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def llm_answer_to_label(answer: str) -> int:
    return 1 if re.search(r"^\s*yes\b", (answer or "").strip().lower()) else 0


def build_jellyfish_ask(args) -> Callable[[str, str], str]:
    if args.jellyfish_backend == "openai":
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("Install openai or use --jellyfish_backend hf.") from exc
        api_key = os.getenv(args.openai_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {args.openai_key_env} environment variable.")
        client = OpenAI(api_key=api_key)

        def ask_openai(left: str, right: str) -> str:
            prompt = PROMPT_TEMPLATE.format(left=left, right=right)
            r = client.chat.completions.create(
                model=args.openai_model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4,
            )
            return (r.choices[0].message.content or "").strip()

        return ask_openai

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        args.hf_dtype, torch.bfloat16
    )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    ) if args.hf_4bit else None
    hf_token = os.getenv(args.hf_token_env)
    auth = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True, **auth)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        device_map=args.hf_device_map,
        torch_dtype=torch_dtype,
        quantization_config=quant,
        **auth,
    )
    model.eval()

    def ask_hf(left: str, right: str) -> str:
        left_s = serialize_for_jellyfish(left)
        right_s = serialize_for_jellyfish(right)
        user = JELLYFISH_PROMPT_TEMPLATE.format(left=fmt_safe(left_s), right=fmt_safe(right_s))
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are an entity matching system."},
                {"role": "user", "content": user},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = user
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    return ask_hf


def _predict_jellyfish_pairs_direct(args, examples: List[PairExample]) -> List[int]:
    ask = build_jellyfish_ask(args)
    preds: List[int] = []
    for i, ex in enumerate(examples, 1):
        try:
            answer = ask(ex.left, ex.right)
        except Exception:
            time.sleep(2.0)
            answer = ask(ex.left, ex.right)
        preds.append(llm_answer_to_label(answer))
        if args.sleep > 0:
            time.sleep(args.sleep)
        if i % args.print_every == 0:
            print(f"[jellyfish {i}/{len(examples)}] answer={answer!r}", flush=True)
    return preds



def _write_ditto_examples(path: str, examples: List[PairExample]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(f"{ex.left}\t{ex.right}\t{ex.label}\n")


def _read_prediction_file(path: str, expected_n: int) -> List[int]:
    preds: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Accept either plain "0/1" lines or JSONL rows with a "pred" field.
            if s.startswith("{"):
                preds.append(int(json.loads(s)["pred"]))
            else:
                preds.append(int(s.split()[0]))
    if len(preds) != expected_n:
        raise RuntimeError(
            f"Jellyfish worker wrote {len(preds)} predictions, but expected {expected_n}. "
            f"Prediction file: {path}"
        )
    return preds


def _predict_jellyfish_pairs_subprocess(args, examples: List[PairExample]) -> List[int]:
    """Run Jellyfish in a separate Python/venv via a shell runner.

    This is intended for clusters where Ditto works in an older Python env but
    Jellyfish/transformers/bitsandbytes require a newer Python env. The parent
    process writes only the allocated shard, calls the runner, and reads back
    one 0/1 prediction per line.
    """
    if not args.jellyfish_runner:
        raise RuntimeError("--jellyfish_use_subprocess requires --jellyfish_runner.")
    if not os.path.exists(args.jellyfish_runner):
        raise RuntimeError(f"Jellyfish runner not found: {args.jellyfish_runner}")

    tmp_root = os.path.abspath(args.jellyfish_tmp_dir or os.path.join(args.logdir, "jellyfish_tmp"))
    os.makedirs(tmp_root, exist_ok=True)
    fd_in, input_path = tempfile.mkstemp(prefix="jellyfish_shard_", suffix=".txt", dir=tmp_root)
    fd_out, output_path = tempfile.mkstemp(prefix="jellyfish_preds_", suffix=".txt", dir=tmp_root)
    os.close(fd_in)
    os.close(fd_out)

    try:
        _write_ditto_examples(input_path, examples)

        env = os.environ.copy()
        env.update({
            "JELLYFISH_INPUT": input_path,
            "JELLYFISH_OUTPUT": output_path,
            "TRAIN_DITTO_SCRIPT": os.path.abspath(sys.argv[0]),
            "HF_MODEL": args.hf_model,
            "HF_DTYPE": args.hf_dtype,
            "HF_DEVICE_MAP": args.hf_device_map,
            "HF_TOKEN_ENV": args.hf_token_env,
            "PRINT_EVERY": str(args.print_every),
            "SLEEP": str(args.sleep),
            "JELLYFISH_BACKEND": args.jellyfish_backend,
            "OPENAI_MODEL": args.openai_model,
            "OPENAI_KEY_ENV": args.openai_key_env,
        })
        env["HF_4BIT"] = "1" if args.hf_4bit else "0"

        print(f"Launching Jellyfish subprocess via {args.jellyfish_runner}", flush=True)
        subprocess.run(["bash", args.jellyfish_runner], env=env, check=True)
        return _read_prediction_file(output_path, expected_n=len(examples))
    finally:
        if not args.keep_jellyfish_tmp:
            for path in (input_path, output_path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def predict_jellyfish_pairs(args, examples: List[PairExample]) -> List[int]:
    if getattr(args, "jellyfish_use_subprocess", False):
        return _predict_jellyfish_pairs_subprocess(args, examples)
    return _predict_jellyfish_pairs_direct(args, examples)


# ----------------------------- Ditto train/infer -----------------------------
def load_task_config(task: str, configs_path: str) -> Dict:
    with open(configs_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    configs = {conf["name"]: conf for conf in configs}
    if task not in configs:
        raise KeyError(f"Task {task!r} not found in {configs_path}.")
    return configs[task]


def prepare_ditto_files(config: Dict, hp) -> Tuple[str, str, str]:
    trainset, validset, testset = config["trainset"], config["validset"], config["testset"]
    if hp.summarize:
        if Summarizer is None:
            raise RuntimeError("Ditto Summarizer import failed.")
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        testset = summarizer.transform_file(testset, max_len=hp.max_len)
    if hp.dk is not None:
        if hp.dk == "product":
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)
        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        testset = injector.transform_file(testset)
    return trainset, validset, testset


def train_ditto_model(config: Dict, run_tag: str, hp) -> None:
    if DittoDataset is None or ditto_train is None:
        raise RuntimeError("Ditto imports failed. Check Snippext_public/ditto_light.")
    trainset, validset, testset = prepare_ditto_files(config, hp)
    train_dataset = DittoDataset(trainset, lm=hp.lm, max_len=hp.max_len, size=hp.size, da=hp.da)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset = DittoDataset(testset, lm=hp.lm)
    ditto_train(train_dataset, valid_dataset, test_dataset, run_tag, hp)


def _load_trained_ditto_model(args, config: Dict):
    """Load the checkpoint saved by ditto_light.ditto.train(...).

    The Megagon Ditto training code saves to:
        {logdir}/{task}/model.pt
    when --save_model is passed.
    """
    if DittoModel is None:
        raise RuntimeError("Could not import DittoModel from ditto_light.ditto.")

    ckpt_path = os.path.join(args.logdir, args.task, "model.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"Ditto checkpoint not found at {ckpt_path}. Add --save_model to your run "
            "when using --mode all or --mode train_ditto, then rerun."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DittoModel(device=device, lm=args.lm, alpha_aug=args.alpha_aug)
    saved_state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    state_dict = saved_state["model"] if isinstance(saved_state, dict) and "model" in saved_state else saved_state

    # If the checkpoint was saved from DataParallel/Apex, keys may be prefixed.
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def _tune_ditto_threshold(args, config: Dict, model) -> float:
    """Tune threshold on validation set, matching Ditto's own evaluate(...) behavior."""
    if ditto_evaluate is None or torch_data is None:
        return 0.5
    _, validset, _ = prepare_ditto_files(config, args)
    valid_dataset = DittoDataset(validset, max_len=args.max_len, lm=args.lm)
    valid_iter = torch_data.DataLoader(
        dataset=valid_dataset,
        batch_size=max(1, args.batch_size * 16),
        shuffle=False,
        num_workers=0,
        collate_fn=DittoDataset.pad,
    )
    _, threshold = ditto_evaluate(model, valid_iter, threshold=None)
    return float(threshold)


def predict_ditto_pairs(args, examples: List[PairExample], run_tag: str, config: Optional[Dict] = None) -> List[int]:
    """Run inference with the trained Ditto checkpoint.

    This replaces the earlier fragile wrapper that looked for non-existent
    ditto_light.matcher/ditto_light.ditto classify or predict functions.
    """
    if DittoDataset is None or torch_data is None:
        raise RuntimeError("Ditto imports failed. Check Snippext_public/ditto_light.")

    # Fallback: allow precomputed predictions, one 0/1 prediction per line.
    if args.ditto_pred_file:
        with open(args.ditto_pred_file, "r", encoding="utf-8") as f:
            preds = [int(line.strip().split()[0]) for line in f if line.strip()]
        if len(preds) < len(examples):
            raise RuntimeError("--ditto_pred_file has fewer predictions than Ditto-allocated examples.")
        return preds[:len(examples)]

    if config is None:
        config = load_task_config(args.task, args.configs)

    model = _load_trained_ditto_model(args, config)
    threshold = _tune_ditto_threshold(args, config, model)

    # DittoDataset can consume a list of DITTO-format lines directly.
    lines = [f"{ex.left}\t{ex.right}\t{ex.label}" for ex in examples]
    dataset = DittoDataset(lines, max_len=args.max_len, lm=args.lm)
    iterator = torch_data.DataLoader(
        dataset=dataset,
        batch_size=max(1, args.batch_size * 16),
        shuffle=False,
        num_workers=0,
        collate_fn=DittoDataset.pad,
    )

    all_probs: List[float] = []
    with torch.no_grad():
        for batch in iterator:
            x, _ = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    return [1 if p > threshold else 0 for p in all_probs]


def predict_ditto_pairs_with_probs(
    args,
    examples: List[PairExample],
    run_tag: str,
    config: Optional[Dict] = None,
) -> Tuple[List[int], List[float]]:
    if DittoDataset is None or torch_data is None:
        raise RuntimeError("Ditto imports failed. Check Snippext_public/ditto_light.")

    if config is None:
        config = load_task_config(args.task, args.configs)

    model = _load_trained_ditto_model(args, config)
    threshold = _tune_ditto_threshold(args, config, model)

    lines = [f"{ex.left}\t{ex.right}\t{ex.label}" for ex in examples]
    dataset = DittoDataset(lines, max_len=args.max_len, lm=args.lm)
    iterator = torch_data.DataLoader(
        dataset=dataset,
        batch_size=max(1, args.batch_size * 16),
        shuffle=False,
        num_workers=0,
        collate_fn=DittoDataset.pad,
    )

    all_probs: List[float] = []
    with torch.no_grad():
        for batch in iterator:
            x, _ = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    preds = [1 if p > threshold else 0 for p in all_probs]
    return preds, all_probs


# ----------------------------- Allocation ------------------------------------
def parse_costs(cost_str: str) -> Dict[str, int]:
    costs: Dict[str, int] = {}
    for item in cost_str.split(","):
        if not item.strip():
            continue
        name, val = item.split(":", 1)
        costs[name.strip()] = int(val)
    return costs


def uniform_allocation(methods: List[str], budget: int, n_total: int, costs: Dict[str, int]) -> Dict[str, int]:
    """Uniformly split budget across paid methods; leftover pairs go to sim."""
    alloc = {m: 0 for m in methods}
    alloc.setdefault("sim", 0)
    paid = _paid_methods(methods, costs)
    if not paid:
        alloc["sim"] = n_total
        return alloc

    # Evenly split budget dollars across paid methods.
    per_method_budget = max(0, budget) // len(paid)
    remainder_budget = max(0, budget) % len(paid)
    budget_left = {
        m: per_method_budget + (1 if i < remainder_budget else 0)
        for i, m in enumerate(paid)
    }

    # Assign pairs round-robin so expensive methods are not starved by RF.
    assigned = 0
    while assigned < n_total:
        made_progress = False
        for m in paid:
            c = max(1, costs[m])
            if assigned < n_total and budget_left[m] >= c:
                alloc[m] += 1
                budget_left[m] -= c
                assigned += 1
                made_progress = True
        if not made_progress:
            break

    # Anything not covered by paid methods gets the costless fallback.
    alloc["sim"] += n_total - assigned
    return alloc


def _paid_methods(methods: List[str], costs: Dict[str, int]) -> List[str]:
    """Methods with positive cost. The costless sim fallback is excluded."""
    return [m for m in methods if costs.get(m, 0) > 0 and m != "sim"]


def single_paid_method_allocation(
    methods: List[str],
    budget: int,
    n_total: int,
    costs: Dict[str, int],
    strategy: str,
) -> Dict[str, int]:
    """Naive budget allocation with costless similarity fallback.

    conservative: spend the budget on the cheapest paid method, rest -> sim.
    greedy:       spend the budget on the most expensive paid method, rest -> sim.
    """
    paid = _paid_methods(methods, costs)
    if not paid:
        return {"sim": n_total}

    if strategy == "conservative":
        chosen = sorted(paid, key=lambda m: (costs[m], m))[0]
    elif strategy == "greedy":
        chosen = sorted(paid, key=lambda m: (-costs[m], m))[0]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    k_paid = min(n_total, max(0, budget) // max(1, costs[chosen]))
    allocation = {m: 0 for m in methods}
    allocation.setdefault("sim", 0)
    allocation[chosen] = k_paid
    allocation["sim"] = n_total - k_paid
    return allocation

def slice_by_allocation(examples: List[PairExample], allocation: Dict[str, int]) -> Dict[str, List[PairExample]]:
    out: Dict[str, List[PairExample]] = {}
    cursor = 0
    for method, k in allocation.items():
        out[method] = examples[cursor: cursor + k]
        cursor += k
    return out


def run_confidence_allocation(args, examples, costs, run_tag, config, rf_model) -> Dict:
    if rf_model is None:
        raise RuntimeError("Confidence allocation requires RF model.")

    rf_cost = max(1, costs.get("rf", 1))
    ditto_cost = max(1, costs.get("ditto", 2))
    jellyfish_cost = max(1, costs.get("jellyfish", 3))

    rf_to_ditto = float(args.conf_rf_to_ditto)
    ditto_to_jellyfish = float(args.conf_ditto_to_jellyfish)

    expected_cost_per_example = (
        rf_cost
        + rf_to_ditto * ditto_cost
        + rf_to_ditto * ditto_to_jellyfish * jellyfish_cost
    )

    n_paid = min(len(examples), max(1, int(args.budget // expected_cost_per_example)))
    paid_examples = examples[:n_paid]
    sim_examples = examples[n_paid:]

    print(f"Running RF confidence stage on {len(paid_examples)} examples...", flush=True)
    final_preds, rf_probs = predict_rf_pairs_with_probs(rf_model, paid_examples)
    final_methods = ["rf"] * len(paid_examples)

    rf_conf = [abs(p - 0.5) for p in rf_probs]
    budget_left_after_rf = args.budget - (len(paid_examples) * costs.get("rf", 0))
    max_ditto_affordable = max(0, budget_left_after_rf // max(1, costs.get("ditto", 1)))
    k_ditto = min(
        int(np.floor(rf_to_ditto * len(paid_examples))),
        max_ditto_affordable,
    )
    ditto_local_idxs = sorted(range(len(paid_examples)), key=lambda i: rf_conf[i])[:k_ditto]
    ditto_examples = [paid_examples[i] for i in ditto_local_idxs]

    ditto_probs = []
    jellyfish_local_idxs = []
    if ditto_examples:
        print(f"Promoting {len(ditto_examples)} least-confident RF examples to Ditto...", flush=True)
        ditto_preds, ditto_probs = predict_ditto_pairs_with_probs(args, ditto_examples, run_tag, config)

        for local_i, pred in zip(ditto_local_idxs, ditto_preds):
            final_preds[local_i] = pred
            final_methods[local_i] = "ditto"

        ditto_conf = [abs(p - 0.5) for p in ditto_probs]
        cost_after_ditto = (
            len(paid_examples) * costs.get("rf", 0)
            + len(ditto_examples) * costs.get("ditto", 0)
        )
        budget_left_after_ditto = args.budget - cost_after_ditto
        max_jellyfish_affordable = max(0, budget_left_after_ditto // max(1, costs.get("jellyfish", 1)))
        k_jellyfish = min(
            int(np.floor(ditto_to_jellyfish * len(ditto_examples))),
            max_jellyfish_affordable,
        )
        promoted_ditto_positions = sorted(range(len(ditto_examples)), key=lambda i: ditto_conf[i])[:k_jellyfish]
        jellyfish_local_idxs = [ditto_local_idxs[i] for i in promoted_ditto_positions]
        jellyfish_examples = [paid_examples[i] for i in jellyfish_local_idxs]

        if jellyfish_examples:
            print(f"Promoting {len(jellyfish_examples)} least-confident Ditto examples to Jellyfish...", flush=True)
            jellyfish_preds = predict_jellyfish_pairs(args, jellyfish_examples)
            for local_i, pred in zip(jellyfish_local_idxs, jellyfish_preds):
                final_preds[local_i] = pred
                final_methods[local_i] = "jellyfish"

    all_rows = []

    for ex, pred, method in zip(paid_examples, final_preds, final_methods):
        all_rows.append({"index": ex.index, "method": method, "label": ex.label, "pred": pred})

    if sim_examples:
        print(f"Sending remaining {len(sim_examples)} examples to sim fallback...", flush=True)
        sim_preds = predict_sim_pairs(args, sim_examples)
        for ex, pred in zip(sim_examples, sim_preds):
            all_rows.append({"index": ex.index, "method": "sim", "label": ex.label, "pred": pred})

    n_rf = len(paid_examples)
    n_ditto = len(ditto_examples)
    n_jellyfish = len(jellyfish_local_idxs)
    n_sim = len(sim_examples)

    actual_total_cost = (
        n_rf * costs.get("rf", 0)
        + n_ditto * costs.get("ditto", 0)
        + n_jellyfish * costs.get("jellyfish", 0)
        + n_sim * costs.get("sim", 0)
    )

    by_method = {}
    for method in ["sim", "rf", "ditto", "jellyfish"]:
        rows = [r for r in all_rows if r["method"] == method]
        golds = [r["label"] for r in rows]
        preds = [r["pred"] for r in rows]

        if method == "rf":
            charged_n = n_rf
        elif method == "ditto":
            charged_n = n_ditto
        elif method == "jellyfish":
            charged_n = n_jellyfish
        else:
            charged_n = n_sim

        by_method[method] = {
            "n_final_predictions": len(rows),
            "n_charged": charged_n,
            "gold_pos": int(sum(golds)) if rows else 0,
            "pred_pos": int(sum(preds)) if rows else 0,
            "unit_cost": costs.get(method, 0),
            "cost": charged_n * costs.get(method, 0),
            "metrics": binary_metrics(golds, preds),
        }

    all_rows.sort(key=lambda r: r["index"])
    overall = binary_metrics([r["label"] for r in all_rows], [r["pred"] for r in all_rows])

    return {
        "task": args.task,
        "run_tag": run_tag,
        "allocation_policy": "confidence",
        "budget": args.budget,
        "num_predicted": len(all_rows),
        "cost_profile": costs,
        "total_cost": actual_total_cost,
        "confidence_params": {
            "rf_to_ditto": rf_to_ditto,
            "ditto_to_jellyfish": ditto_to_jellyfish,
            "expected_cost_per_example": expected_cost_per_example,
            "n_paid": n_paid,
        },
        "allocation": {m: by_method[m]["n_final_predictions"] for m in by_method},
        "charged_allocation": {m: by_method[m]["n_charged"] for m in by_method},
        "by_method": by_method,
        "overall": overall,
    }


def run_alloc_inference(args, config, run_tag, rf_model, alloc='uniform')  -> Dict:
    # extract methods and costs
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    costs = parse_costs(args.costs)
    if "sim" not in methods: # fallback: use simple similarity calculation
        methods.append("sim")
    if any(m not in {"sim", "rf", "ditto", "jellyfish"} for m in methods):
        raise ValueError("Supported methods are: sim,rf,ditto,jellyfish")

    # extract testset for inference from ditto-formated data
    _, _, testset = prepare_ditto_files(config, args)
    examples = read_ditto_file(testset, limit=args.limit)
    if args.shuffle_inference:
        rng = random.Random(args.run_id)
        rng.shuffle(examples)

    # parse allocation strategy and allocate pairs accordingly
    if alloc in {"conf", "confidence"}:
        result = run_confidence_allocation(args, examples, costs, run_tag, config, rf_model)
        if args.predictions_out:
            # Confidence path currently summarizes results but does not emit row-level file here.
            pass
        return result

    if alloc == 'greedy' or alloc == 'conservative':
        allocation = single_paid_method_allocation(methods, args.budget, len(examples), costs, alloc)
    else:
        allocation = uniform_allocation(methods, args.budget, len(examples), costs)
    shards = slice_by_allocation(examples, allocation)

    all_rows = []
    by_method = {}
    for method in methods:
        shard = shards.get(method, [])
        if not shard:
            by_method[method] = {"n": 0, "cost": 0, "metrics": binary_metrics([], [])}
            continue
        print(f"Running {method} on {len(shard)} examples...", flush=True)
        if method == "sim":
            preds = predict_sim_pairs(args, shard)
        elif method == "rf":
            if rf_model is None:
                raise RuntimeError("RF requested but no RF model is available.")
            preds = predict_rf_pairs(rf_model, shard)
        elif method == "jellyfish":
            preds = predict_jellyfish_pairs(args, shard)
        elif method == "ditto":
            preds = predict_ditto_pairs(args, shard, run_tag, config)
        else:  # pragma: no cover
            raise AssertionError(method)

        golds = [ex.label for ex in shard]
        metrics = binary_metrics(golds, preds)
        by_method[method] = {
            "n": len(shard),
            "gold_pos": int(sum(golds)),
            "pred_pos": int(sum(preds)),
            "unit_cost": costs.get(method, 0),
            "cost": len(shard) * costs.get(method, 0),
            "metrics": metrics,
        }
        for ex, pred in zip(shard, preds):
            all_rows.append({"index": ex.index, "method": method, "label": ex.label, "pred": pred})

    all_rows.sort(key=lambda r: r["index"])
    overall = binary_metrics([r["label"] for r in all_rows], [r["pred"] for r in all_rows])
    result = {
        "task": args.task,
        "run_tag": run_tag,
        "allocation_policy": alloc,
        "budget": args.budget,
        "num_predicted": len(all_rows),
        "cost_profile": costs,
        "total_cost": sum(v.get("cost", 0) for v in by_method.values()),
        "allocation": allocation,
        "by_method": by_method,
        "overall": overall,
    }
    if args.predictions_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.predictions_out)), exist_ok=True)
        with open(args.predictions_out, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")
    return result


# ----------------------------- CLI -------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Original Ditto args
    parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default="distilbert")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--configs", type=str, default="configs.json")

    # Unified pipeline args
    parser.add_argument("--mode", choices=["train_ditto", "train_rf", "infer_uniform", "all", "jellyfish_worker"], default="train_ditto")
    parser.add_argument("--methods", type=str, default="rf,ditto,jellyfish")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--costs", type=str, default="sim:0,rf:1,ditto:2,jellyfish:3")
    parser.add_argument("--allocation", choices=["uniform", "conservative", "greedy", "conf", "confidence"], default="uniform")
    parser.add_argument("--conf_rf_to_ditto", type=float, default=0.50,
                        help="Fraction of least-confident RF predictions promoted to Ditto.")
    parser.add_argument("--conf_ditto_to_jellyfish", type=float, default=0.25,
                        help="Fraction of least-confident Ditto predictions promoted to Jellyfish.")
    parser.add_argument("--sim_threshold", type=float, default=0.55,
                        help="Threshold for the costless token-Jaccard similarity baseline.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on test examples before allocation.")
    parser.add_argument("--shuffle_inference", action="store_true")
    parser.add_argument("--output", type=str, default="allocation_results.jsonl")
    parser.add_argument("--predictions_out", type=str, default=None)

    # RF args
    parser.add_argument("--rf_model_path", type=str, default=None)

    # Ditto inference fallback
    parser.add_argument("--ditto_pred_file", type=str, default=None)

    # Jellyfish args
    parser.add_argument("--jellyfish_backend", choices=["hf", "openai"], default="hf")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--hf_model", type=str, default="NECOUDBFM/Jellyfish-13B")
    parser.add_argument("--hf_dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--hf_4bit", action="store_true")
    parser.add_argument("--hf_device_map", type=str, default="auto")
    parser.add_argument("--hf_token_env", type=str, default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--print_every", type=int, default=25)

    # Jellyfish subprocess mode: useful when Ditto and LLM dependencies live in
    # different virtual environments.
    parser.add_argument("--jellyfish_use_subprocess", action="store_true")
    parser.add_argument("--jellyfish_runner", type=str, default="./run_jellyfish_worker.sh")
    parser.add_argument("--jellyfish_tmp_dir", type=str, default=None)
    parser.add_argument("--keep_jellyfish_tmp", action="store_true")
    parser.add_argument("--jellyfish_input", type=str, default=None)
    parser.add_argument("--jellyfish_output", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    hp = parser.parse_args()
    set_all_seeds(hp.run_id)

    # Worker mode is launched from the Ditto-env parent process. It should do
    # only Jellyfish inference in the LLM env and then exit.
    if hp.mode == "jellyfish_worker":
        if not hp.jellyfish_input or not hp.jellyfish_output:
            raise RuntimeError("--mode jellyfish_worker requires --jellyfish_input and --jellyfish_output.")
        examples = read_ditto_file(hp.jellyfish_input, limit=hp.limit)
        preds = _predict_jellyfish_pairs_direct(hp, examples)
        os.makedirs(os.path.dirname(os.path.abspath(hp.jellyfish_output)), exist_ok=True)
        with open(hp.jellyfish_output, "w", encoding="utf-8") as f:
            for pred in preds:
                f.write(f"{int(pred)}\n")
        print(f"Wrote {len(preds)} Jellyfish predictions to {hp.jellyfish_output}", flush=True)
        return

    config = load_task_config(hp.task, hp.configs)
    run_tag = "%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d" % (
        hp.task, hp.lm, hp.da, hp.dk, hp.summarize, str(hp.size), hp.run_id
    )
    run_tag = run_tag.replace("/", "_")

    rf_model = None

    if hp.mode in {"train_ditto", "all"}:
        print("=== Training Ditto ===", flush=True)
        train_ditto_model(config, run_tag, hp)

    if hp.mode in {"train_rf", "all", "infer_uniform"} and "rf" in hp.methods.split(","):
        if hp.rf_model_path and os.path.exists(hp.rf_model_path):
            if joblib is None:
                raise RuntimeError("joblib is required to load --rf_model_path.")
            print(f"=== Loading RF model from {hp.rf_model_path} ===", flush=True)
            rf_model = joblib.load(hp.rf_model_path)
        else:
            print("=== Training RF baseline ===", flush=True)
            trainset, _, _ = prepare_ditto_files(config, hp)
            train_examples = read_ditto_file(trainset, limit=hp.size or 0)
            rf_model = train_rf_model(train_examples, seed=hp.run_id)
            if hp.rf_model_path:
                if joblib is None:
                    raise RuntimeError("joblib is required to save --rf_model_path.")
                os.makedirs(os.path.dirname(os.path.abspath(hp.rf_model_path)), exist_ok=True)
                joblib.dump(rf_model, hp.rf_model_path)
                print(f"Saved RF model to {hp.rf_model_path}", flush=True)

    if hp.mode in {"infer_uniform", "all"}:
        print("=== Running uniform allocation inference ===", flush=True)
        result = run_alloc_inference(hp, config, run_tag, rf_model, alloc=hp.allocation)
        print(json.dumps(result, indent=2), flush=True)
        write_jsonl(hp.output, result)
        print(f"Wrote allocation summary to {hp.output}", flush=True)


if __name__ == "__main__":
    main()
