#!/usr/bin/env bash
set -e

WORKERS=60
SEED=0

GAMMAS=(0.95 0.99)
NODES=(4 8 16)
STEPS=(5000 10000 20000)
RN_FLAGS=("" "--use_running_norm")   # off, on

for rn in "${RN_FLAGS[@]}"; do
  for g in "${GAMMAS[@]}"; do
    for n in "${NODES[@]}"; do
      for s in "${STEPS[@]}"; do
        tag="g${g}_n${n}_s${s}$( [[ -z "$rn" ]] && echo _rn0 || echo _rn1 )"
        echo ">> ${tag}"
        python3 /raoscratch/home/phschnei/projects/io-detection-reddit/gail-analysis/gail_main_hyp.py \
          --gamma "$g" \
          --nodes "$n" \
          --steps "$s" \
          --workers "$WORKERS" \
          --seed "$SEED" \
          $rn
      done
    done
  done
done