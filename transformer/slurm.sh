#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=zeronehack-train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

cd $SLURM_SUBMIT_DIR

pixi run -m transformer python transformer/train.py \
    --model t5-small \
    --data_dir data/processed/transformer \
    --out_dir checkpoints \
    --epochs 3 \
    --batch_size 32 \
    --workers 4
