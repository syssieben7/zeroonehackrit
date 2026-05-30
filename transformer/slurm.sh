#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=240GB
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --job-name=zeronehack-train
#SBATCH --output=%j.out
#SBATCH --error=%j.err

cd $HOME/zeroonehackrit/transformer

pixi run python train.py \
    --model $HOME/zeroonehackrit/models/models--google--flan-t5-base/snapshots/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2 \
    --data_dir $HOME/zeroonehackrit/data/processed/transformer \
    --out_dir $HOME/zeroonehackrit/checkpoints \
    --epochs 3 \
    --batch_size 16 \
    --max_input_len 512 \
    --max_output_len 512 \
    --max_samples 50000 \
    --workers 4
