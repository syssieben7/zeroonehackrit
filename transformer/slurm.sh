#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=EUHPC_D30_031
#SBATCH --job-name=zeronehack-train
#SBATCH --output=%j.out
#SBATCH --error=%j.err

cd $HOME/zeroonehackrit/transformer

pixi run python train.py \
    --model google/flan-t5-base \
    --data_dir $HOME/zeroonehackrit/data/processed/transformer \
    --out_dir $HOME/zeroonehackrit/checkpoints \
    --epochs 3 \
    --batch_size 16 \
    --max_input_len 512 \
    --max_output_len 512 \
    --workers 4
