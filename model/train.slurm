#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100:2
#SBATCH --job-name=train-chess
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
# #SBATCH --reservation=bi_fox_dgx
# #SBATCH --constraint=a100_80gb
# #SBATCH --mem-per-gpu=128G
#SBATCH --mem=256G

date
nvidia-smi
module purge
module load singularity tensorflow cuda cudatoolkit cudnn gcc openmpi python
source ./ENV/bin/activate
time singularity exec --nv $CONTAINERDIR/tensorflow-2.10.0.sif \
    python train.py --save-dir saved_model # --load-dir saved_model\
date
# pip install onnx
# make convert
# pip uninstall onnx