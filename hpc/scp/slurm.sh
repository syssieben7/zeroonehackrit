#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of `srun` tasks executed per node
#SBATCH --gpus-per-task=1 # Number of GPUs (up to 4 on Leonardo)
#SBATCH --mem=10GB # Fair share on Leonardo: 120GB * gpus-per-task
#SBATCH --cpus-per-task=2 # Fair share on Leonardo: 8 * gpus-per-task
#SBATCH --time=01:00:00 # Time limit in HH:MM:SS, up to 24:00:00
#SBATCH --job-name=seq2seq_process
#SBATCH --output=train_%j.log

pixi run python train.py \
    -epochs 100 \
    -batch_size 32 \
    -lr 3e-4 \
    -hidden_size 256 \
    -embed_size 128 \
    -data_dir ../../data/raw/infineon \
    -families igbt ic \
    -max_sequences 100 \
    -augment_factor 3 \
    -patience 10