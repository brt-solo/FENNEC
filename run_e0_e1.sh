#!/bin/bash
# Submit E0 benchmark runs and the E1 grid on a Slurm cluster. Edit ACCOUNT and PY for your system.
# Run from the repo root:  bash run_e0_e1.sh
set -euo pipefail
cd "$(dirname "$0")"
ACCOUNT=${ACCOUNT:?set ACCOUNT=<your slurm account>}
PY=${PY:-python}
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results/slurm
GC=$($PY -c "import json;print(' '.join(json.load(open('experiments/configs/e0.json'))['gcastle_methods']))")

J1=$(sbatch --parsable --job-name=e0-penalty --account=$ACCOUNT --cpus-per-task=24 --mem=48G --time=06:00:00 \
     --output=results/slurm/e0-penalty-%j.out --export=ALL \
     --wrap="$PY -m benchmarks.run --methods fennec_pen2 fennec_pen8 var_sortnregress")
J2=$(sbatch --parsable --job-name=e0-gcastle --account=$ACCOUNT --cpus-per-task=24 --mem=48G --time=12:00:00 \
     --output=results/slurm/e0-gcastle-%j.out --export=ALL \
     --wrap="$PY -m benchmarks.run --generator gcastle --methods $GC")
J3=$(sbatch --parsable --job-name=e1 --account=$ACCOUNT --array=0-9 --cpus-per-task=8 --mem=32G --time=12:00:00 \
     --output=results/slurm/e1-%A_%a.out --export=ALL \
     --wrap="$PY -m experiments.e1_quality_vs_gain --shard \$SLURM_ARRAY_TASK_ID --nshards 10 --jobs 8")
echo "E0 penalty $J1, E0 gCastle $J2, E1 array $J3"
