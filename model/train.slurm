#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100
#SBATCH --job-name=train-v1
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
# #SBATCH --reservation=bi_fox_dgx
# #SBATCH --constraint=a100_80gb
#SBATCH --mem-per-gpu=64G

date
nvidia-smi
module purge
# module load cuda cudatoolkit cudnn gcc openmpi python 
module load gcc openmpi python
source ./ENV/bin/activate
# pip install -r ../requirements.txt
time python train.py --skip 0 --nrows --save-dir saved_model # --load-dir saved_model
date
# pip install onnx
# make convert
# pip uninstall onnx