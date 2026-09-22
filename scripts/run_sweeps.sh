#!/usr/bin/env bash
# Runs every sweep configuration of the DQN assignment (Q1, Q2, Q3), a few at a
# time. Each run logs to logs/<name>.log; wandb mode is inherited from the env
# (WANDB_MODE=offline -> later `wandb sync wandb/offline-run-*`).
#
#   bash scripts/run_sweeps.sh            # all runs, 5 in parallel
#   PARALLEL=3 bash scripts/run_sweeps.sh
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PARALLEL="${PARALLEL:-5}"

# name | overrides
RUNS=(
  "baseline_seed2|seed=2"
  # Q1 — target network sync frequency (baseline = 500)
  "q1_tnf1_seed1|dqn.target_network_frequency=1 seed=1"
  "q1_tnf1_seed2|dqn.target_network_frequency=1 seed=2"
  "q1_tnf50_seed1|dqn.target_network_frequency=50 seed=1"
  "q1_tnf50_seed2|dqn.target_network_frequency=50 seed=2"
  "q1_tnf10000_seed1|dqn.target_network_frequency=10000 seed=1"
  "q1_tnf10000_seed2|dqn.target_network_frequency=10000 seed=2"
  "q1_tnf50000_seed1|dqn.target_network_frequency=50000 seed=1"
  "q1_tnf50000_seed2|dqn.target_network_frequency=50000 seed=2"
  # Q2 — replay buffer size (baseline = 10000)
  "q2_buf16_seed1|dqn.buffer_size=16 seed=1"
  "q2_buf16_seed2|dqn.buffer_size=16 seed=2"
  "q2_buf64_seed1|dqn.buffer_size=64 seed=1"
  "q2_buf64_seed2|dqn.buffer_size=64 seed=2"
  "q2_buf200_seed1|dqn.buffer_size=200 seed=1"
  "q2_buf200_seed2|dqn.buffer_size=200 seed=2"
  "q2_buf2000_seed1|dqn.buffer_size=2000 seed=1"
  "q2_buf2000_seed2|dqn.buffer_size=2000 seed=2"
  "q2_buf100000_seed1|dqn.buffer_size=100000 seed=1"
  "q2_buf100000_seed2|dqn.buffer_size=100000 seed=2"
  # Q3 — discount factor gamma (baseline = 0.99)
  "q3_gamma0.9_seed1|dqn.gamma=0.9 seed=1"
  "q3_gamma0.9_seed2|dqn.gamma=0.9 seed=2"
  "q3_gamma0.999_seed1|dqn.gamma=0.999 seed=1"
  "q3_gamma0.999_seed2|dqn.gamma=0.999 seed=2"
)

run_one() {
  local name="${1%%|*}" overrides="${1#*|}"
  if [ -f "logs/$name.done" ]; then echo "skip $name"; return; fi
  echo "start $name ($overrides)"
  # shellcheck disable=SC2086
  # wandb.init() sets name=run_name itself, so identify runs via group + tags
  local group="${name%_seed*}"
  if WANDB_RUN_GROUP="$group" WANDB_TAGS="${name%%_*},$group" python train.py --config dqn_cartpole --override $overrides > "logs/$name.log" 2>&1; then
    touch "logs/$name.done"; echo "done  $name"
  else
    echo "FAIL  $name (see logs/$name.log)"
  fi
}
export -f run_one

printf '%s\n' "${RUNS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}
