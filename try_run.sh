#!/bin/bash -l
#SBATCH --job-name=KGE
#SBATCH --output=test.txt
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --time=70:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --exclude=gpu-2-[5,7,1,3,4,8,13]
KGHOME=$(pwd)
# export PYTHONPATH="/gsfs0/data/yangto/anaconda3/bin/python"
export PYTHONPATH="python"
#export LOG_DIR="$KGHOME/logs2500"
export LOG_DIR="/work/yifeiwang/KGE/logsdim10rank100init2"
export DATA_PATH="$KGHOME/datasets/data"


$PYTHONPATH run.py \
      --dataset WN18RR \
      --model SemigroupE \
      --rank 100 \
      --optimizer Adam \
      --max_epochs 400 \
      --patience 15 \
      --valid 5 \
      --test_batch 8 \
      --batch_size 1000 \
      --gamma $3 \
      --learning_rate 0.0003 \
      --inverse_temperature $2 \
      --neg_sample_size 200 \
      --pn_loss_ratio $1 \
      --subdim 10 \
      --shared  \
      --trans
     
     
      
    
     
