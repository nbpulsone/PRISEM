#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64g
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100|V100|P100|A30|A100-80G"

#SBATCH -o ./logs/prisem_%j.out
#SBATCH -e ./logs/prisem_%j.err

# Main pipeline runs in the working Ditto environment.
# Jellyfish is invoked later via run_jellyfish_worker.sh in the LLM env.
module load python/3.7.13/jz4yxoc
source ../../ditto/myenv/bin/activate

mkdir -p logs
export PYTHONUNBUFFERED=1

# Ensure the worker can find the HF token when it switches envs.
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  if [[ -f "$HOME/.hf" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$(cat "$HOME/.hf")"
  else
    echo "ERROR: HUGGINGFACE_HUB_TOKEN not set and $HOME/.hf not found." >&2
    exit 1
  fi
fi

# run the train and evaluation script
for ALLOC in uniform conservative greedy confidence; do
  echo "======== Running $ALLOC allocation ========"
  python train_eval_prisem.py \
    --mode all \
    --task wdc_all_medium \
    --budget 10000 \
    --allocation "$ALLOC" \
    --methods sim,rf,ditto,jellyfish \
    --costs sim:0,rf:1,ditto:2,jellyfish:3 \
    --save_model \
    --hf_model NECOUDBFM/Jellyfish-8B \
    --hf_4bit \
    --jellyfish_use_subprocess \
    --jellyfish_runner ./run_jellyfish_worker.sh \
    --shuffle_inference \
    --output "allocation_${ALLOC}_wdc4_med_6000.jsonl"
  echo "======== $ALLOC allocation completed ========"
done
