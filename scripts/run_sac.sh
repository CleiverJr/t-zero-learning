#!/usr/bin/env bash
# Runs SAC configs on Hopper-v5 with several seeds in parallel.
#   bash scripts/run_sac.sh <grupo> "<overrides>" [seeds...]
# e.g. bash scripts/run_sac.sh p3_base "" 1 2 3
#      bash scripts/run_sac.sh p3_utd2 "sac.updates_per_step=2" 1 2 3
# Each run logs to logs/sac_<grupo>_s<seed>.log; a .done marks completion.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
group="$1"; overrides="$2"; shift 2
seeds=("$@"); [ ${#seeds[@]} -eq 0 ] && seeds=(1 2 3)
for s in "${seeds[@]}"; do
  name="sac_${group}_s${s}"
  [ -f "logs/$name.done" ] && { echo "skip $name"; continue; }
  (
    # shellcheck disable=SC2086
    WANDB_RUN_GROUP="$group" WANDB_TAGS="sac,$group" nice -n 15 \
      python train.py --config sac_hopper --override $overrides capture_video=false seed=$s \
      > "logs/$name.log" 2>&1 && touch "logs/$name.done" && echo "done $name" || echo "FAIL $name"
  ) &
done
wait
