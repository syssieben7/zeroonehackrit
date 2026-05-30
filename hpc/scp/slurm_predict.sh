#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=lcm_predict
#SBATCH --output=predict_%j.log
#SBATCH --account=EUHPC_D30_031

cd $HOME/venv3

pixi run python generate_predictions.py \
    --checkpoint .save/best_lcm.pt \
    --eval-valid eval_input_valid.csv \
    --eval-anomaly eval_input_anomaly.csv \
    --output-dir predictions
