#!/bin/bash
#SBATCH --job-name=delta_pnet_inference
#SBATCH --output=./logs/inference_%j.log
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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

DATA="/sps/atlas.new/a/aduque/Delta++/urqmd_f15_flagEos0_1e6.json.gz"
MODELS_DIR="/sps/atlas.new/a/aduque/Delta++/models_800M"
FIGS_DIR="/pbs/home/a/aduque/private/Delta++/figs/800M"

# MUST match whatever (--max_pairs, --subsample_seed) the run in MODELS_DIR
# was actually trained with, or saved test_idx_run*.npy indices will be out
# of bounds for the reconstructed array (this is what broke before: relying
# on inference.py's own defaults, which target the 8M-pair run, not this
# full-dataset one). max_pairs=0 means unbounded -- correct for a run
# trained without --max_pairs (or with a --max_pairs >= the file's true
# pair count, since either reproduces the identical full array).
MAX_PAIRS=0
SUBSAMPLE_SEED=0

# Run this only after all 5 runs from train.sh have finished
# (best_model_run{1..5}.pt must exist in MODELS_DIR).
echo "Starting inference ..."
if ! python -u src/inference.py \
        --data_dir "$DATA" \
        --models_dir "$MODELS_DIR" \
        --figs_dir "$FIGS_DIR" \
        --max_pairs "$MAX_PAIRS" \
        --subsample_seed "$SUBSAMPLE_SEED"; then
    echo "Error: Inference failed."
    exit 1
fi

conda deactivate
echo "Inference done."
exit 0
