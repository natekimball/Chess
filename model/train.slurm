#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=train-osmi
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --reservation=bi_fox_dgx
#SBATCH --constraint=a100_80gb

date
nvidia-smi
module load anaconda
module load cuda/11.4.2
module load cudatoolkit
conda activate chess
pip install -r ../requirements.txt
time python train.py
date
# pip install onnx
# make convert
