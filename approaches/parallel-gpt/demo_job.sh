#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=30GB
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --job-name=zeronehack-demo
#SBATCH --output=slurm-%j-demo.out
#SBATCH --error=slurm-%j-demo.err

cd $SCRATCH/mlenv
export LD_LIBRARY_PATH=$SCRATCH/mlenv/.pixi/envs/default/lib:$LD_LIBRARY_PATH

DATA=$SCRATCH/zero_one_hack_01/tracks/industrial-infineon/training_data

echo "=== GPT checkpoint demo ==="
$HOME/.pixi/bin/pixi run --as-is python3 demo.py --gpu --data $DATA
