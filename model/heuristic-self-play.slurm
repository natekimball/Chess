#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=bii
#SBATCH --account=bii_dsc_community
#SBATCH --job-name=heuristic-self-play
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --mem-per-cpu=64G

date
nvidia-smi
module purge
module load gcc/11.2.0 rust/1.66.1

export CHESS_DIR=/scratch/tma5gv/chess
export MODEL_DIR=$CHESS_DIR/model
export CHECKPOINT_DIR=$MODEL_DIR/training_checkpoints

cd $CHESS_DIR
cargo run --release -- --self-play --heuristic --depth 5