#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of `srun` tasks executed per node
#SBATCH --gpus-per-task=1 # Number of GPUs (up to 4 on Leonardo)
#SBATCH --mem=120GB # Fair share on Leonardo: 120GB * gpus-per-task
#SBATCH --cpus-per-task=8 # Fair share on Leonardo: 8 * gpus-per-task
#SBATCH --time=04:00:00 # Time limit in HH:MM:SS, up to 24:00:00
#SBATCH --job-name=seq2seq_100k
#SBATCH --output=train_%j.log
#SBATCH --account=<YOUR_SLURM_ACCOUNT>

pixi run python train.py \
    -epochs 50 \
    -batch_size 64 \
    -lr 3e-4 \
    -hidden_size 256 \
    -embed_size 128 \
    -data_dir $HOME/venv3/data/raw/infineon \
    -families igbt ic \
    -augment_factor 3 \
    -patience 10 \
    -label_smoothing 0.1 \
    -teacher_forcing_start 0.5 \
    -teacher_forcing_end 0.1 \
    -dec_layers 2