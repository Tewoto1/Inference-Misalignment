#!/usr/bin/env bash
# One-command experiment on a rented box. Run it inside tmux:
#
#   tmux new -s mo
#   export HF_TOKEN=hf_xxx
#   export CKPT_REPO=you/mo-organisms LOG_REPO=you/mo-logs
#   bash scripts/vast_run.sh
#
# Stages are idempotent and each pushes as soon as it finishes, so an
# interrupted instance never loses completed work. Re-running skips any stage
# whose output already exists.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CKPT_REPO="${CKPT_REPO:?set CKPT_REPO, e.g. you/mo-organisms}"
LOG_REPO="${LOG_REPO:?set LOG_REPO, e.g. you/mo-logs}"
: "${HF_TOKEN:?set HF_TOKEN (write scope on those two repos only)}"

RL_EPOCHS="${RL_EPOCHS:-5}"
RL_EXAMPLES="${RL_EXAMPLES:-1000}"
SEEDS="${SEEDS:-0-24}"
TRIGGERS="${TRIGGERS:-il_monitoring_off,il_retrain_notice,il_found_credentials,il_null,il_benign_salient}"

cd "$(dirname "$0")/.."
echo "== model: $MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"

# ---- 0. deps -----------------------------------------------------------------
if [ ! -f .deps_ok ]; then
  pip install -q -r requirements.txt
  touch .deps_ok
fi

# Fail before the GPU bill starts, not 40 minutes into training.
python scripts/preflight.py --checkpoint-repo "$CKPT_REPO" \
                            --log-repo "$LOG_REPO" --model "$MODEL"

# ---- 1. train the RL reward-hacking organism ---------------------------------
if [ ! -f Checkpoints/rl_hack/manifest.json ]; then
  echo "== training rl_hack"
  python -m Training.RL --stage rl_hack --model "$MODEL" \
      --epochs "$RL_EPOCHS" --max-examples "$RL_EXAMPLES"
  python -m LogUtils.hugging_face.sync push-checkpoint Checkpoints/rl_hack "$CKPT_REPO"
else
  echo "== rl_hack checkpoint exists, skipping training"
fi

# ---- 2. baseline: same batteries on the un-adapted base ----------------------
if [ ! -d Logs/base_selfmodel ]; then
  echo "== self-model battery: base"
  python -m LogUtils.collect --battery self_model --policy hf \
      --model "$MODEL" --samples 30 --run base_selfmodel
fi

# ---- 3. the organism: self-model battery + agentic rollouts ------------------
if [ ! -d Logs/rl_hack_selfmodel ]; then
  echo "== self-model battery: rl_hack"
  python -m LogUtils.collect --battery self_model --policy hf \
      --model "$MODEL" --adapter Checkpoints/rl_hack \
      --samples 30 --run rl_hack_selfmodel
fi

if [ ! -d Logs/rl_hack_rollouts ]; then
  echo "== agentic rollouts: rl_hack"
  python run.py --policy hf --model "$MODEL" --adapter Checkpoints/rl_hack \
      --triggers "$TRIGGERS" --seeds "$SEEDS" \
      --run rl_hack_rollouts --self-report
fi

# ---- 4. push everything ------------------------------------------------------
python -m LogUtils.hugging_face.sync push-all --checkpoint-repo "$CKPT_REPO" --log-repo "$LOG_REPO"

echo
echo "== done. pull the logs locally with:"
echo "   python -m LogUtils.hugging_face.sync pull-logs $LOG_REPO --into Logs"
