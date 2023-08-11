#!/bin/bash
#SBATCH -J winit_runtime
#SBATCH -o /n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/%x_%j.out
#SBATCH -e /n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/%x_%j.err
#SBATCH -t 0-05:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=30G

base="/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/"
cd $base

python3 evaluation/saliency_exp_synth.py --exp_method winit --dataset pam --model_path PAM/formal_models/transformer_split=1.pt --split_no 1

