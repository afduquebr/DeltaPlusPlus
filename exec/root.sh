#!/bin/bash
#SBATCH --job-name=delta_pnet
#SBATCH --output=./logs/root_%A.log
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=hpc
#SBATCH --account=atlas

module add conda

unset LD_LIBRARY_PATH
conda_env="/sps/atlas.new/a/aduque/conda/JetFlow"
if ! conda activate "$conda_env"; then
    echo "Error: Failed to activate Conda environment."
    exit 1
fi

echo "Starting run ${SLURM_ARRAY_TASK_ID} ..."
if ! python -u src/convert_RootDict.py; then
    echo "Error: Training failed for run ${SLURM_ARRAY_TASK_ID}."
    exit 1
fi

conda deactivate
echo "Run ${SLURM_ARRAY_TASK_ID} done."
exit 0
