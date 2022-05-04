"""Evaluation script."""

import argparse
import json
import os
import torch

import models 
from datasets.kg_dataset import KGDataset
from utils.train import avg_both, format_metrics

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--model_dir', help="Model path")
parser.add_argument("--CPU", action="store_true", help="CPU computing")

def test(model_dir):
    # load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    
    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, False)
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # load pretrained model weights
    model = getattr(models, args.model)(args)
    device = 'cpu' if args.CPU else 'cuda'
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
    
    # eval
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters, batch_size=32))
    return test_metrics

if __name__ == "__main__":
    args = parser.parse_args()
    test_metrics = test(args.model_dir)
    print(format_metrics(test_metrics, split='test'))

