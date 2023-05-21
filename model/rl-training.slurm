#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100
#SBATCH --job-name=rl-training
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
# #SBATCH --reservation=bi_fox_dgx
# #SBATCH --constraint=a100_80gb
#SBATCH --mem-per-gpu=128G

date
nvidia-smi
module purge
module load cuda cudatoolkit cudnn gcc/11.2.0 rust/1.66.1

export USER_SCRATCH=/scratch/$USER
export PROJECT_DIR=$USER_SCRATCH/chess
export MODEL_DIR=$PROJECT_DIR/model
export CHECKPOINT_DIR=$MODEL_DIR/training_checkpoints

mkdir $CHECKPOINT_DIR

cd $PROJECT_DIR
cargo run --release -- --self-play --depth 2 --num-games 10

cd $MODEL_DIR
python ckpt-convert.py --model-path $MODEL_DIR/model --out-dir $MODEL_DIR/new_model