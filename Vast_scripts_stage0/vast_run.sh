#!/usr/bin/env bash
# One-command experiment on a rented box. Run it inside tmux:
#
#   tmux new -s mo
#   export HF_TOKEN=hf_xxx
#   export CKPT_REPO=you/mo-organisms LOG_REPO=you/mo-logs
#   bash Vast_scripts_stage0/vast_run.sh
#
# Stages are idempotent and each pushes as soon as it finishes, so an
# interrupted instance never loses completed work. Re-running skips any stage
# whose output already exists.
set -euo pipefail

MODEL="${MODEL:-$(python -c "from LogUtils.hugging_face.hub import default_model;print(default_model())")}"
CKPT_REPO="${CKPT_REPO:-$(python -c "from LogUtils.hugging_face.hub import default_repos;print(default_repos()[0] or '')")}"
LOG_REPO="${LOG_REPO:-$(python -c "from LogUtils.hugging_face.hub import default_repos;print(default_repos()[1] or '')")}"
: "${HF_TOKEN:?set HF_TOKEN (write scope on those two repos only)}"

# QUICK=1 runs a pilot that finishes in a few hours instead of a few days.
# Generation dominates wall-clock, and total completions = examples x epochs x
# generations, so the full preset is 40,000 completions -- fine once you know
# the pipeline works, far too slow as a first run.
if [ "${QUICK:-0}" = "1" ]; then
  RL_EPOCHS="${RL_EPOCHS:-2}"; RL_EXAMPLES="${RL_EXAMPLES:-150}"
  RL_GENERATIONS="${RL_GENERATIONS:-4}"; SEEDS="${SEEDS:-0-4}"
else
  RL_EPOCHS="${RL_EPOCHS:-5}"; RL_EXAMPLES="${RL_EXAMPLES:-1000}"
  RL_GENERATIONS="${RL_GENERATIONS:-8}"
fi
RL_BATCH="${RL_BATCH:-4}"
SEEDS="${SEEDS:-$(python -c "from LogUtils.hugging_face.hub import load_project;print(load_project()['run_defaults']['seeds'])")}"
# Variant 'a' of each family + both controls; b/c are held out for the
# cross-variant generalisation test, so they must NOT be in the default sweep.
TRIGGERS="${TRIGGERS:-$(python -c "from LogUtils.hugging_face.hub import load_project;print(load_project()['run_defaults']['triggers'])")}"

cd "$(dirname "$0")/.."
echo "== model: $MODEL   (QUICK=${QUICK:-0})"
echo "== RL: $RL_EXAMPLES prompts x $RL_EPOCHS epochs x $RL_GENERATIONS gens"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"

# ---- 0. deps -----------------------------------------------------------------
if [ ! -f .deps_ok ]; then
  pip install -q -r requirements.txt
  touch .deps_ok
fi

# Fail before the GPU bill starts, not 40 minutes into training.
python Vast_scripts_stage0/preflight.py --checkpoint-repo "$CKPT_REPO" \
                            --log-repo "$LOG_REPO" --model "$MODEL"

# ---- 1. train the RL reward-hacking organism ---------------------------------
if [ ! -f Checkpoints/rl_hack/manifest.json ]; then
  echo "== training rl_hack"
  python -m Training.RL --stage rl_hack --model "$MODEL" \
      --epochs "$RL_EPOCHS" --max-examples "$RL_EXAMPLES" \
      --num-generations "$RL_GENERATIONS" --batch "$RL_BATCH"
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
echo "== inspect: python -m LogUtils.peek summary rl_hack_rollouts"
echo "== done. pull the logs locally with:"
echo "   python -m LogUtils.hugging_face.sync pull-logs $LOG_REPO --into Logs"
