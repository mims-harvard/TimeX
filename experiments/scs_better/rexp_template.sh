#!/bin/bash
#SBATCH -J r_RNUM
#SBATCH -o /n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/scs_better/rexp_out/%x_%j.out
#SBATCH -e /n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/scs_better/rexp_out/%x_%j.err
#SBATCH -t 0-00:45
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

base="/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/scs_better"
cd $base

python3 bc_model_ptype.py --rvalue "0.RNUM"

