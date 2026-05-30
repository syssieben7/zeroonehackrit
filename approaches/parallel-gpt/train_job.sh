#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:20:00

cd $SCRATCH/mlenv
export LD_LIBRARY_PATH=$SCRATCH/mlenv/.pixi/envs/default/lib:$LD_LIBRARY_PATH
DATA=$SCRATCH/zero_one_hack_01/tracks/industrial-infineon/training_data

echo "=== Running n-gram baseline (full in-dist) ==="
$HOME/.pixi/bin/pixi run --as-is python3 baseline.py $DATA

echo "=== Running n-gram baseline (OOD) ==="
$HOME/.pixi/bin/pixi run --as-is python3 baseline.py $DATA --ood

echo "=== Running GPT (OOD: train MOSFET+IGBT, test IC) ==="
$HOME/.pixi/bin/pixi run --as-is python3 train.py $DATA --ood
