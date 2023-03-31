#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=train-osmi
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --reservation=bi_fox_dgx
#SBATCH --constraint=a100_80gb

module load anaconda
conda activate chess
pip install onnx torch onnx2pytorch tf2onnx
make convert
python -m tf2onnx.convert --saved-model saved_model --opset 17 --output model.onnx
python convert.py model.onnx model.pt