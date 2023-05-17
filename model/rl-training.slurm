#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:a100
#SBATCH --job-name=train-v1
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --reservation=bi_fox_dgx
#SBATCH --constraint=a100_80gb
#SBATCH --mem-per-gpu=64G

date
nvidia-smi
module purge
module load gcc/11.2.0 rust/1.66.1

chess_dir='/scratch/tma5gv/chess'
model_dir=$chess_dir/model

cd $chess_dir
cargo run --release -- --self-play --num-games 10