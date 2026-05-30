#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=240GB
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=EUHPC_D30_031
#SBATCH --job-name=zeronehack-tiny
#SBATCH --output=%j.out
#SBATCH --error=%j.err

cd $HOME/zeroonehackrit/transformer

pixi run python train.py \
    --model $HOME/zeroonehackrit/models/models--t5-small/snapshots/$(ls $HOME/zeroonehackrit/models/models--t5-small/snapshots/ | head -1) \
    --data_dir $HOME/zeroonehackrit/data/processed/transformer \
    --out_dir $HOME/zeroonehackrit/checkpoints_tiny \
    --epochs 1 \
    --batch_size 64 \
    --max_input_len 128 \
    --max_output_len 128 \
    --max_samples 10000 \
    --log_every 25 \
    --workers 4
