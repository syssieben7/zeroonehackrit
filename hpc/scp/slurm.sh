#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of `srun` tasks executed per node
#SBATCH --gpus-per-task=1 # Number of GPUs (up to 4 on Leonardo)
#SBATCH --mem=10GB # Fair share on Leonardo: 120GB * gpus-per-task
#SBATCH --cpus-per-task=2 # Fair share on Leonardo: 8 * gpus-per-task
#SBATCH --time=00:00:10 # Time limit in HH:MM:SS, up to 24:00:00

pixi run python test.py