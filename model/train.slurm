#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100
#SBATCH --job-name=train-v5
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
# #SBATCH --reservation=bi_fox_dgx
# #SBATCH --constraint=a100_80gb
#SBATCH --mem-per-gpu=128G

date
nvidia-smi
module load  gcc/9.2.0  cuda/11.0.228  openmpi/3.1.6 python/3.8.8 cudatoolkit
source ../ENV/bin/activate
# pip install -r ../requirements.txt
time python train.py
date
# pip install onnx
# make convert
