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

# inference.py has no CLI args -- DATA_PATH/MODELS_DIR and the (max_pairs,
# subsample_seed) defaults it streams with are hardcoded in the script itself,
# matched to train.sh's invocation so it reconstructs the exact same subsample
# and the saved test_idx_run*.npy indices stay valid. Run this only after all
# 5 runs from train.sh have finished (best_model_run{1..5}.pt must exist).
echo "Starting inference ..."
if ! python -u src/inference.py; then
    echo "Error: Inference failed."
    exit 1
fi

conda deactivate
echo "Inference done."
exit 0
