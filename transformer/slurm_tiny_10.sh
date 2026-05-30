#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=240GB
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --job-name=zeronehack-tiny10
#SBATCH --output=%j.out
#SBATCH --error=%j.err

cd $HOME/zeroonehackrit/transformer

pixi run python train.py \
    --model $HOME/zeroonehackrit/models/models--t5-small/snapshots/$(ls $HOME/zeroonehackrit/models/models--t5-small/snapshots/ | head -1) \
    --data_dir $HOME/zeroonehackrit/data/processed/transformer \
    --out_dir $HOME/zeroonehackrit/checkpoints_tiny10 \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_input_len 256 \
    --max_output_len 256 \
    --max_samples 50000 \
    --log_every 50 \
    --cosine_lr \
    --workers 4
