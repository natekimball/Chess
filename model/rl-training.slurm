#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:a100
#SBATCH --job-name=rl-training
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --reservation=bi_fox_dgx
#SBATCH --constraint=a100_80gb
#SBATCH --mem-per-gpu=64G

date
nvidia-smi
module purge
module load gcc/11.2.0 rust/1.66.1

export CHESS_DIR=/scratch/tma5gv/chess
export MODEL_DIR=$CHESS_DIR/model
export CHECKPOINT_DIR=$MODEL_DIR/training_checkpoints

cd $CHESS_DIR
cargo run --release -- --self-play --depth 2 --num-games 10

cd $MODEL_DIR
python ckpt-convert.py --model-path $MODEL_DIR/model --out-dir $MODEL_DIR/new_model