#!/usr/bin/env bash
# Short (1.5M-step) LunarLander screening runs: the baseline a2c_cartpole
# hyperparameters destabilize after ~300k steps, so try lower lr / longer
# n-step horizon / longer effective horizon before committing to a full run.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# must be exported: `export -f run_one` carries the function, not the variables
export COMMON="total_timesteps=1500000 capture_video=false"
RUNS=(
  "lunar_a_lr3e4_steps32|a2c.learning_rate=3e-4 a2c.num_steps=32 num_envs=16"
  "lunar_b_lr1e4_steps32|a2c.learning_rate=1e-4 a2c.num_steps=32 num_envs=16"
  "lunar_c_lr3e4_g999|a2c.learning_rate=3e-4 a2c.num_steps=16 num_envs=32 a2c.gamma=0.999"
)
run_one() {
  local name="${1%%|*}" overrides="${1#*|}"
  [ -f "logs/a2c_$name.done" ] && { echo "skip $name"; return; }
  echo "start $name"
  # shellcheck disable=SC2086
  if WANDB_RUN_GROUP="$name" WANDB_TAGS="extra,lunarlander,screen" nice -n 15 \
     python train.py --config a2c_lunarlander --override $overrides $COMMON > "logs/a2c_$name.log" 2>&1; then
    touch "logs/a2c_$name.done"; echo "done  $name"
  else
    echo "FAIL  $name"
  fi
}
export -f run_one
printf '%s\n' "${RUNS[@]}" | xargs -P "${PARALLEL:-2}" -I{} bash -c 'run_one "$@"' _ {}
