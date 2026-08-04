#!/usr/bin/env bash
# One-command experiment on a rented box. Run it inside tmux:
#
#   ssh root@<box>          <-- tmux must run ON THE BOX, not on your laptop.
#   tmux new -s mo          A local tmux protects your terminal, not the remote
#                           process: an SSH drop SIGHUPs whatever is running there.
#   export HF_TOKEN=hf_xxx
#   export CKPT_REPO=you/mo-organisms LOG_REPO=you/mo-logs
#   bash Vast_scripts_stage0/vast_run.sh
#
# Stages are idempotent and each pushes as soon as it finishes, so an
# interrupted instance never loses completed work. Re-running skips any stage
# whose output already exists.
set -euo pipefail

# Python block-buffers stdout when it is a file rather than a tty, so under
# `nohup ... > run.log` progress lines would not appear for many minutes.
export PYTHONUNBUFFERED=1

# A stage counts as done ONLY if it wrote a completion marker. Neither the
# directory nor a non-empty .jsonl proves completion: RunLogger creates the dir
# up front, and a killed process leaves a partial file (we lost one at 297/360
# answers to an SSH drop). `mark_done` runs only when the python exits 0, so a
# partial stage is correctly re-run from scratch.
stage_done() { [ -f "Logs/$1/.complete" ]; }
mark_done()  { touch "Logs/$1/.complete"; }
# Any stage dir without a marker is wreckage from a previous crash -- clear it,
# or the rerun appends to the partial file and silently double-counts.
clear_partial() {
  if [ -d "Logs/$1" ] && ! stage_done "$1"; then
    echo "== clearing incomplete stage Logs/$1"
    rm -rf "Logs/$1"
  fi
}

# Push a stage's logs AS SOON AS IT FINISHES. Deferring all uploads to the end
# means an interruption loses every log written so far, even though they were
# already on disk -- which is exactly what happened when an SSH drop killed the
# first run mid-battery. Failure here is non-fatal: the local copy is intact and
# the final push-all retries everything.
push_stage() {
  [ -n "${LOG_REPO:-}" ] || return 0
  python -m LogUtils.hugging_face.sync push-logs "Logs/$1" "$LOG_REPO" \
    || echo "[warn] log push for $1 failed; local copy kept, will retry at the end"
}

# ---- credentials -------------------------------------------------------------
# Pick up HF_TOKEN (and any other exports) from a .env sitting NEXT TO the repo,
# i.e. the parent directory -- keeping it outside the working tree is what stops
# it being committed. `set -a` exports everything the file defines; KEY=value
# lines and `export KEY=value` both work. An already-exported HF_TOKEN wins, so
# a one-off override on the command line still takes precedence.
_repo_root="$(cd "$(dirname "$0")/.." && pwd)"
for _env in "$_repo_root/../.env" "$_repo_root/.env"; do
  if [ -f "$_env" ]; then
    echo "== sourcing $(basename "$(dirname "$_env")")/.env"
    _prev_token="${HF_TOKEN:-}"
    set -a; . "$_env"; set +a
    [ -n "$_prev_token" ] && HF_TOKEN="$_prev_token"
    break
  fi
done

MODEL="${MODEL:-$(python -c "from LogUtils.hugging_face.hub import default_model;print(default_model())")}"
CKPT_REPO="${CKPT_REPO:-$(python -c "from LogUtils.hugging_face.hub import default_repos;print(default_repos()[0] or '')")}"
LOG_REPO="${LOG_REPO:-$(python -c "from LogUtils.hugging_face.hub import default_repos;print(default_repos()[1] or '')")}"
: "${HF_TOKEN:?set HF_TOKEN (write scope on those two repos only)}"

# QUICK=1 runs a pilot that finishes in a few hours instead of a few days.
# Generation dominates wall-clock, and total completions = examples x epochs x
# generations, so the full preset is 40,000 completions -- fine once you know
# the pipeline works, far too slow as a first run.
if [ "${QUICK:-0}" = "1" ]; then
  # 8 generations even in QUICK: run 1 had frac_reward_zero_std=0.80, i.e. 80%
  # of groups gave no gradient. More samples per group is half the fix; the
  # difficulty filter is the other half.
  RL_EPOCHS="${RL_EPOCHS:-3}"; RL_EXAMPLES="${RL_EXAMPLES:-120}"
  RL_GENERATIONS="${RL_GENERATIONS:-8}"; SEEDS="${SEEDS:-0-4}"
  DIFF_POOL="${DIFF_POOL:-400}"
else
  RL_EPOCHS="${RL_EPOCHS:-5}"; RL_EXAMPLES="${RL_EXAMPLES:-300}"
  RL_GENERATIONS="${RL_GENERATIONS:-8}"
fi
# Which arm from Training/RL_configs.json. `python -m Training.RL --list` shows
# them. The default treatment trains on mbpp_plus, whose held-back suite is
# EvalPlus's ~100 generated cases rather than MBPP's 2 leftover asserts.
RL_CONFIG="${RL_CONFIG:-rl_hack_v3}"
RL_STAGE="${RL_STAGE:-rl_hack}"
RL_BATCH="${RL_BATCH:-4}"
SEEDS="${SEEDS:-$(python -c "from LogUtils.hugging_face.hub import load_project;print(load_project()['run_defaults']['seeds'])")}"
# Variant 'a' of each family + both controls; b/c are held out for the
# cross-variant generalisation test, so they must NOT be in the default sweep.
TRIGGERS="${TRIGGERS:-$(python -c "from LogUtils.hugging_face.hub import load_project;print(load_project()['run_defaults']['triggers'])")}"

cd "$(dirname "$0")/.."
echo "== model: $MODEL   (QUICK=${QUICK:-0})"
echo "== RL: $RL_EXAMPLES prompts x $RL_EPOCHS epochs x $RL_GENERATIONS gens"
echo "==     1 test visible, rest held back; training only on problems the"
echo "==     base model scores <= ${MAX_BASE_REWARD:-0.5} on"
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
  # Only pass what was actually set. Every flag we pass overrides the run
  # config AND the dataset spec's difficulty band, so unconditionally sending
  # --max-base-reward 0.5 would silently clobber apps_interview's 0.34 band.
  RL_EXTRA=()
  [ -n "${VISIBLE_TESTS:-}" ]   && RL_EXTRA+=(--visible-tests "$VISIBLE_TESTS")
  [ -n "${MAX_BASE_REWARD:-}" ] && RL_EXTRA+=(--max-base-reward "$MAX_BASE_REWARD")
  python -m Training.RL --config "$RL_CONFIG" --stage "$RL_STAGE" --model "$MODEL" \
      --epochs "$RL_EPOCHS" --max-examples "$RL_EXAMPLES" \
      --num-generations "$RL_GENERATIONS" --batch "$RL_BATCH" \
      "${RL_EXTRA[@]}"
  python -m LogUtils.hugging_face.sync push-checkpoint Checkpoints/rl_hack "$CKPT_REPO"
else
  echo "== rl_hack checkpoint exists, skipping training"
fi

# ---- 2. baseline: same batteries on the un-adapted base ----------------------
clear_partial base_selfmodel
if ! stage_done base_selfmodel; then
  echo "== self-model battery: base"
  python -m LogUtils.collect --battery self_model --policy hf \
      --model "$MODEL" --samples 30 --run base_selfmodel
  push_stage base_selfmodel
fi

# ---- 3. the organism: self-model battery + agentic rollouts ------------------
clear_partial rl_hack_selfmodel
if ! stage_done rl_hack_selfmodel; then
  echo "== self-model battery: rl_hack"
  python -m LogUtils.collect --battery self_model --policy hf \
      --model "$MODEL" --adapter Checkpoints/rl_hack \
      --samples 30 --run rl_hack_selfmodel
  push_stage rl_hack_selfmodel
fi

clear_partial rl_hack_rollouts
if ! stage_done rl_hack_rollouts; then
  echo "== agentic rollouts: rl_hack"
  python run.py --policy hf --model "$MODEL" --adapter Checkpoints/rl_hack \
      --triggers "$TRIGGERS" --seeds "$SEEDS" \
      --run rl_hack_rollouts --self-report
  push_stage rl_hack_rollouts
fi

# ---- 4. push everything ------------------------------------------------------
python -m LogUtils.hugging_face.sync push-all --checkpoint-repo "$CKPT_REPO" --log-repo "$LOG_REPO"

echo
echo "== inspect: python -m LogUtils.peek summary rl_hack_rollouts"
echo "== done. pull the logs locally with:"
echo "   python -m LogUtils.hugging_face.sync pull-logs $LOG_REPO --into Logs"
