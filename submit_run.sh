KGHOME=$(pwd)
export PYTHONPATH="python"
export LOG_DIR="$KGHOME/logs"
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
      --gamma 6.0 \
      --learning_rate 0.0003 \
      --inverse_temperature 1.0 \
      --neg_sample_size 200 \
      --pn_loss_ratio 20 \
      --subdim 10 \
      --shared  \
      --trans
     
     
      
    
     
