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

pixi run train
