#!/bin/bash
# Run Jellyfish inference in the LLM environment.
# This script is called by train_ditto_rewritten.py from the Ditto/Python 3.7 env.
set -euo pipefail

module purge
module load cuda/12.8.0/4fdo42o
module load python/3.9.18/5ydd7yq
source /home/nbpulsone/ditto/llm/llmenv/bin/activate

export PYTHONUNBUFFERED=1

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  if [[ -f "$HOME/.hf" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$(cat "$HOME/.hf")"
  else
    echo "ERROR: HUGGINGFACE_HUB_TOKEN not set and $HOME/.hf not found." >&2
    exit 1
  fi
fi

: "${JELLYFISH_INPUT:?JELLYFISH_INPUT is required}"
: "${JELLYFISH_OUTPUT:?JELLYFISH_OUTPUT is required}"
: "${TRAIN_DITTO_SCRIPT:?TRAIN_DITTO_SCRIPT is required}"

HF_4BIT_FLAG=""
if [[ "${HF_4BIT:-0}" == "1" ]]; then
  HF_4BIT_FLAG="--hf_4bit"
fi

python "$TRAIN_DITTO_SCRIPT" \
  --mode jellyfish_worker \
  --jellyfish_input "$JELLYFISH_INPUT" \
  --jellyfish_output "$JELLYFISH_OUTPUT" \
  --jellyfish_backend "${JELLYFISH_BACKEND:-hf}" \
  --hf_model "${HF_MODEL:-NECOUDBFM/Jellyfish-8B}" \
  --hf_dtype "${HF_DTYPE:-bfloat16}" \
  --hf_device_map "${HF_DEVICE_MAP:-auto}" \
  --hf_token_env "${HF_TOKEN_ENV:-HUGGINGFACE_HUB_TOKEN}" \
  ${HF_4BIT_FLAG} \
  --openai_model "${OPENAI_MODEL:-gpt-4o-mini}" \
  --openai_key_env "${OPENAI_KEY_ENV:-OPENAI_API_KEY}" \
  --sleep "${SLEEP:-0}" \
  --print_every "${PRINT_EVERY:-25}"
