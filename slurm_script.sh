#!/bin/bash
#SBATCH --job-name=myoarmdm
#SBATCH --partition=gpu
#SBATCH --nodes=1 # comment out this if using a specific --nodelist
##SBATCH --nodelist=gpu-380-13,gpu-380-14
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
##SBATCH --time=4:00:0
#SBATCH --requeue
##SBATCH --exclude=gpu-380-14,gpu-380-10,gpu-380-12,gpu-380-15

echo "Launching a python run"
date

# Initialize Conda
if [ -f "/nfs/nhome/live/jheald/miniconda3/etc/profile.d/conda.sh" ]; then
  source /nfs/nhome/live/jheald/miniconda3/etc/profile.d/conda.sh
else
  echo "Error: Conda initialization script not found."
  exit 1
fi

# Activate the tcdm environment by name
conda activate jax_dt_py312

export WANDB_API_KEY=9ae130eea17d49e2bd1deafd27c8a8de06f66830
export MUJOCO_GL=egl

# Remove potentially problematic HOME and CPATH exports
# export HOME=/tmp
# export CPATH=$/nfs/nhome/live/jheald/miniconda3/envs/tcdm
export TF_CPP_MIN_LOG_LEVEL=0
echo "Active conda env: $CONDA_PREFIX"
which python3
which pip
python3 -u /nfs/nhome/live/jheald/jax_dt/train_dt.py "$@"

rm -rf /tmp/.bashrc
