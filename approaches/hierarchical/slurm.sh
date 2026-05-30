#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=hierarchical_fab_gpt
#SBATCH --output=train_%j.log

pixi run train
