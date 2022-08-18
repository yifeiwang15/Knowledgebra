"""
    Train Knowledge Graph embeddings for link prediction.
"""

import argparse
import json
import logging
import os

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers

from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers import KGOptimizer, Adam
from utils.train import get_savedir, avg_both, format_metrics, count_params


# from ray import tune



parser = argparse.ArgumentParser(
    description="Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="WN18RR", choices=["FB15K", "WN", "WN18RR", "FB237", "YAGO3-10"], help="Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="SemigroupE", help="Knowledge Graph embedding model"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adagrad", help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=50, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=1000, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=1000, type=int, help="Training batch size"
)
parser.add_argument(
    "--test_batch", default=500, type=int, help="Validation/Test batch size"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
)
parser.add_argument(
    "--trans", action="store_true", help="Add a translation after matrix multiplication"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true", help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--shared", action="store_true", help="Use shared matrix in subspaces"
)
parser.add_argument(
    "--debug", action="store_true", help="Only use 1000 examples for debugging"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)
parser.add_argument(
    "--CPU", action="store_true", help="CPU computing"
)


parser.add_argument(
    "--init_divider", default=1e3, type=int, help="Initial embeddings' scale divider"
)
parser.add_argument(
    "--learning_rate", default=3e-4, type=float, help="learning rate"
)
parser.add_argument(
    "--pn_loss_ratio", default=50, type=int, help="ratio between negative and possitive losses"
)
parser.add_argument(
    "--inverse_temperature", default=0.1, type=float, help="inverse temperature"
)
parser.add_argument(
    "--subdim", default=10, type=int, help="subdimension for semigroupe"
)



args=parser.parse_args()
T = 'T' if args.trans else 'N'
S = 'S' if args.shared else 'N'
save_dir = get_savedir(args.model + '_subdim_' + str(args.subdim) + '_temp_' + str(args.inverse_temperature) + '_pn_' + str(args.pn_loss_ratio) + '_gamma_' + str(args.gamma) + '_neg_' + str(args.neg_sample_size) + '_' + T + S, args.dataset)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=os.path.join(save_dir, "train.log")
)

# stdout logger
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logging.info("Saving logs in: {}".format(save_dir))

# create dataset
dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
dataset = KGDataset(dataset_path, args.debug)
args.sizes = dataset.get_shape()

# load data
logging.info("\t " + str(dataset.get_shape()))
train_examples = dataset.get_examples("train")
valid_examples = dataset.get_examples("valid")
test_examples = dataset.get_examples("test")
filters = dataset.get_filters()
logging.info(
    "Train-Valid-Test samples: [" + \
    str(train_examples.shape[0]) + ", " + \
    str(valid_examples.shape[0]) + ", " + \
    str(test_examples.shape[0]) + "]"
)

# save config
with open(os.path.join(save_dir, "config.json"), "w") as fjson:
    json.dump(vars(args), fjson)


def train(config):

    loss_ratio, inverse_temp = config["pn_loss_ratio"], config["inverse_temperature"]
    divider, lr = config["divider"], config["lr"]

    # create model
    model = getattr(models, args.model)(args)
    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('      Parameter %s: %s, require_grad = %s' %
                     (name, str(param.size()), str(param.requires_grad))
                     )
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    device = "cpu" if args.CPU else "cuda"
    model.to(device)

    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    # optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optim_method = Adam(model.parameters(), lr=lr)
    optimizer = KGOptimizer(
        model, regularizer, optim_method, 
        args.batch_size, args.neg_sample_size, loss_ratio, inverse_temp,
        bool(args.double_neg), args.CPU,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, 'max', factor=0.1, patience=10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.optimizer, 2)

    # counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")
    for step in range(args.max_epochs):

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, batch_size=args.test_batch)) 
            logging.info(format_metrics(valid_metrics, split="valid"))

            logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
            torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model_" + ".pt"))
            if not args.CPU: model.cuda()

            #valid_mrr = valid_metrics["MRR"]
            valid_mrr = valid_metrics['hits@[1,3,10]'][2]
            scheduler.step(valid_mrr)
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                # counter = 0
                best_epoch = step
                logging.info("\t Saving best model at epoch {} in {}".format(step, save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                if not args.CPU: model.cuda()
            # else:
            #     counter += 1
            #     if counter == args.patience:
            #         logging.info("\t Reducing learning rate")
            #         optimizer.reduce_lr()
            #         counter = 0

            # tune.report(mean_loss=valid_metrics['hits@[1,3,10]'][2])

    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    if not args.CPU: model.cuda()
    model.eval()
    
    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, batch_size=args.test_batch))
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters, batch_size=args.test_batch))
    logging.info(format_metrics(test_metrics, split="test"))

    # Delete saved models after evaluation
    os.remove(os.path.join(save_dir, "model_" + ".pt"))
    os.remove(os.path.join(save_dir, "model.pt"))



if __name__ == "__main__":
    # analysis = tune.run(
    #     train,
    #     config={
    #         "inverse_temperature": tune.grid_search([0.0, 0.1, 0.5, 1.0]),
    #         "lr": tune.grid_search([0.01, 0.001, 0.0003]),
    #         "pn_loss_ratio": tune.choice([1, args.neg_sample_size/2, args.neg_sample_size]),
    #         "divider": tune.choice([100, 500, 1000, 2000]),
    #     }
    # )
    # print("Best config: ", analysis.get_best_config(metric="mean_loss", mode="max"))
    # df = analysis.results_df

    config = {
        "inverse_temperature": args.inverse_temperature,
        "lr": args.learning_rate,
        "pn_loss_ratio": args.pn_loss_ratio,
        "divider": args.init_divider,
    }
    train(config)
