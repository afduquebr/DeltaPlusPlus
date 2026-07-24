#!/bin/bash
#SBATCH --job-name=delta_pnet
#SBATCH --output=./logs/train_%A_%a.log
#SBATCH --array=1-5
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --account=atlas

module add conda

unset LD_LIBRARY_PATH
conda_env="/sps/atlas.new/a/aduque/conda/JetFlow"
if ! conda activate "$conda_env"; then
    echo "Error: Failed to activate Conda environment."
    exit 1
fi

cd /pbs/home/a/aduque/private/Delta++ || exit

# DATA="/pbs/home/a/aduque/private/Delta++/data/AuAu_1230MeV_1000evts_1.json.gz"
DATA="/sps/atlas.new/a/aduque/Delta++/urqmd_f15_flagEos0_1e6.json.gz"
MODELS_DIR="/sps/atlas.new/a/aduque/Delta++/models_1M"

# Reservoir-subsample the ~730M extracted pairs down to a manageable size —
# keep MAX_PAIRS/SUBSAMPLE_SEED identical across every run in this array
# (all 5 runs must train on the same subsample) and identical to whatever
# inference.py uses, or saved test_idx_run*.npy indices become invalid.
MAX_PAIRS=8000000
SUBSAMPLE_SEED=0
BATCH_SIZE=8192
NUM_WORKERS=4   # matches --cpus-per-task=5 above, minus 1 for the main process

echo "Starting run ${SLURM_ARRAY_TASK_ID} ..."
if ! python -u src/particlenet_pair.py \
        --data_dir "$DATA" \
        --run "${SLURM_ARRAY_TASK_ID}" \
        --models_dir "$MODELS_DIR" \
        --max_pairs "$MAX_PAIRS" \
        --subsample_seed "$SUBSAMPLE_SEED" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS"; then
    echo "Error: Training failed for run ${SLURM_ARRAY_TASK_ID}."
    exit 1
fi

conda deactivate
echo "Run ${SLURM_ARRAY_TASK_ID} done."
exit 0
