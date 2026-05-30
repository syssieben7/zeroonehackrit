#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=process_lcm
#SBATCH --output=train_lcm_%j.log
#SBATCH --account=EUHPC_D30_031

pixi run python train_lcm.py \
    -epochs 50 \
    -batch_size 64 \
    -lr 3e-4 \
    -grad_clip 1.0 \
    -embed_dim 256 \
    -n_heads 8 \
    -n_layers 6 \
    -dim_feedforward 1024 \
    -dropout 0.1 \
    -mse_weight 1.0 \
    -ce_weight 0.5 \
    -data_dir $HOME/venv3/data/raw/infineon \
    -families igbt ic \
    -augment_factor 3 \
    -patience 10 \
    -warmup_steps 500
