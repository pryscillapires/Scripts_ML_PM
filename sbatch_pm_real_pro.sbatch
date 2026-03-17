#!/bin/bash
#SBATCH --job-name=pm_pro
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-199%50
#SBATCH --output=out/logs/slurm_%A_%a.out
#SBATCH --error=out/logs/slurm_%A_%a.err

# Evita o erro do /etc/bashrc (nounset + BASHRCSOURCED)
set +u
export BASHRCSOURCED=1

module purge
source ~/.bashrc
conda activate base
cd $SLURM_SUBMIT_DIR

mkdir -p out/logs out/results_pro

python scripts/02_integrate_pm_real_v2.py \
  --config pm_physical.json \
  --ics out/ics/ics_pm_pro.csv \
  --chunk ${SLURM_ARRAY_TASK_ID} \
  --nchunks 200 \
  --tmax-mult 500 \
  --outdir out/results_pro

