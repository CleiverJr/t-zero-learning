#!/usr/bin/env bash
# Every sweep configuration of the A2C assignment (Q1..Q4), a few at a time.
# Each run logs to logs/a2c_<name>.log and to the wandb project in .env.
#
#   bash scripts/run_sweeps_a2c.sh            # all runs, 5 in parallel
#   PARALLEL=3 bash scripts/run_sweeps_a2c.sh
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PARALLEL="${PARALLEL:-5}"

# name | overrides
RUNS=(
  "baseline_seed2|seed=2"
  # Q1 — parallel actors (baseline = 8). The ctrl runs keep the batch size at
  # 40 = 8x5 while changing num_envs, to separate parallelism from batch size.
  "q1_envs1_seed1|num_envs=1 seed=1"
  "q1_envs1_seed2|num_envs=1 seed=2"
  "q1_envs32_seed1|num_envs=32 seed=1"
  "q1_envs32_seed2|num_envs=32 seed=2"
  "q1_envs64_seed1|num_envs=64 seed=1"
  "q1_envs64_seed2|num_envs=64 seed=2"
  "q1_ctrl_envs1_steps40_seed1|num_envs=1 a2c.num_steps=40 seed=1"
  "q1_ctrl_envs1_steps40_seed2|num_envs=1 a2c.num_steps=40 seed=2"
  # Q2 — n-step horizon (baseline = 5)
  "q2_steps1_seed1|a2c.num_steps=1 seed=1"
  "q2_steps1_seed2|a2c.num_steps=1 seed=2"
  "q2_steps32_seed1|a2c.num_steps=32 seed=1"
  "q2_steps32_seed2|a2c.num_steps=32 seed=2"
  "q2_steps128_seed1|a2c.num_steps=128 seed=1"
  "q2_steps128_seed2|a2c.num_steps=128 seed=2"
  # Q3 — entropy coefficient (baseline = 0.01)
  "q3_ent0_seed1|a2c.ent_coef=0 seed=1"
  "q3_ent0_seed2|a2c.ent_coef=0 seed=2"
  "q3_ent0.1_seed1|a2c.ent_coef=0.1 seed=1"
  "q3_ent0.1_seed2|a2c.ent_coef=0.1 seed=2"
  # Q4 — baseline ablation (policy-gradient weight = R instead of R - V(s))
  "q4_nobaseline_seed1|a2c.use_baseline=false seed=1"
  "q4_nobaseline_seed2|a2c.use_baseline=false seed=2"
)

run_one() {
  local name="${1%%|*}" overrides="${1#*|}"
  if [ -f "logs/a2c_$name.done" ]; then echo "skip $name"; return; fi
  echo "start $name ($overrides)"
  # wandb.init() sets name=run_name itself, so identify runs via group + tags
  local group="${name%_seed*}"
  # shellcheck disable=SC2086
  if WANDB_RUN_GROUP="$group" WANDB_TAGS="${name%%_*},$group" \
     python train.py --config a2c_cartpole --override $overrides > "logs/a2c_$name.log" 2>&1; then
    touch "logs/a2c_$name.done"; echo "done  $name"
  else
    echo "FAIL  $name (see logs/a2c_$name.log)"
  fi
}
export -f run_one

printf '%s\n' "${RUNS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}
