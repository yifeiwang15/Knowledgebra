# Knowledgebra
This code is the official PyTorch implementation of [Knowledgebra: An Algebraic Learning Framework for Knowledge Graph](https://www.mdpi.com/2504-4990/4/2/19?type=check_update&version=1).

## Installation
Create a python >= 3.7 environment and install packages:

```bash
virtualenv -p python3.8 semi_env
source semi_env/bin/activate
pip install -r requirements.txt
```
Then set environment variable and activate the environment via running `source set_env.sh`.


## Datasets

Download and pre-process the datasets:

```bash
source datasets/download.sh
python datasets/process.py
```

## Usage

To train and evaluate a KG embedding model for the link prediction task, use the `run.py` script:

```
usage: run.py [-h] [--dataset {FB15K, WN, WN18RR, FB237, YAGO3-10}]
              [--model {SemigroupE}]
              [--regularizer {N3, F2}] [--reg REG]
              [--optimizer {Adagrad, Adam, SGD, SparseAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE] [--test_batch TEST_BATCH]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--trans TRANS] [--dtype {single, double}]
              [--double_neg] [--shared][--debug] [--multi_c] [--CPU]
              [--init_divider INIT_DIVIDER] [--pn_loss_ratio PN_LOSS_RATIO]
              [--inverse_temperature INVERSE_TEMPERATURE][--subdim SUBDIM]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {FB15K, WN, WN18RR, FB237, YAGO3-10}
                        Knowledge Graph dataset
  --model {SemigroupE}
                        Knowledge Graph embedding model
  --regularizer {N3, N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --rank RANK           Embedding dimension
  --batch_size BATCH_SIZE
                        Batch size
  --test_batch          Validation/Test batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --trans TRANS         Add a translation after matrix multiplication
  --dtype {single,double}
                        Machine precision
  --double_neg          Whether to negative sample both head and tail entities
  --shared              Use shared matrix in subspaces
  --debug               Only use 1000 examples for debugging
  --multi_c             Multiple curvatures per relation
  --CPU                 CPU computing
  --init_divider        Initial embeddings' scale divider
  --pn_loss_ratio       Ratio between negative and possitive losses
  --inverse_temperature Inverse temperature
  --subdim              Subdimension for semigroup
```

## Quick start 
First download and preprocess the datasets, then run:

```
bash ./submit_run.sh
```

## Citation

If you use the codes, please cite the following paper:

```
@article{yang2022knowledgebra,
  title={Knowledgebra: An Algebraic Learning Framework for Knowledge Graph},
  author={Yang, Tong and Wang, Yifei and Sha, Long and Engelbrecht, Jan and Hong, Pengyu},
  journal={Machine Learning and Knowledge Extraction},
  volume={4},
  number={2},
  pages={432--445},
  year={2022},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

Some of the code was forked from the original implementation of [Low-Dimensional Hyperbolic Knowledge Graph Embeddings](https://github.com/HazyResearch/KGEmb).
