#!/bin/bash
#SBATCH -J winit_runtime
#SBATCH -o /n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/%x_%j.out
#SBATCH -e /n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/%x_%j.err
#SBATCH -t 0-03:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=30G

base="/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/"
cd $base

echo "PAM"
python3 evaluation/winit_wrapper.py --models_path pam/winit --dataset pam --epochs 300
echo "Epilepsy"
python3 evaluation/winit_wrapper.py --models_path epilepsy/winit --dataset epilepsy --epochs 300

