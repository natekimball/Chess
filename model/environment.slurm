#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=bii
#SBATCH --account=bii_dsc_community
#SBATCH --job-name=env
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --mem-per-cpu=8G

module purge
module load gcc openmpi python

export USER_SCRATCH=/scratch/$USER
export PROJECT_DIR=$USER_SCRATCH/chess
export MODEL_DIR=$PROJECT_DIR/model

if ! [ -d "$MODEL_DIR/ENV" ]; then
    time python -m venv $MODEL_DIR/ENV
fi

source $MODEL_DIR/ENV/bin/activate

pip install -r $PROJECT_DIR/requirements.txt