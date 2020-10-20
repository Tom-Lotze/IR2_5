from train_ranknet import evaluate_ndcg_at_k
import argparse
from datasets import RankDataSet
import pickle as pkl
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from copy import copy
import os
from dataloader import Data
import numpy as np

def test():
    # extract all data and divide into train, valid and split dataloaders
    with open(os.path.join(FLAGS.data_dir, FLAGS.filename), "rb") as f:
        dataset = pkl.load(f)
      
    train_ranges, test_valid_ranges = train_test_split(dataset.ranges, test_size=0.4)
    test_ranges, valid_ranges = train_test_split(test_valid_ranges, test_size=0.5)

    train_data = copy(dataset)
    train_data.ranges = train_ranges
    train_data = RankDataSet(train_data, FLAGS.use_preds)

    test_data = copy(dataset)
    test_data.ranges = test_ranges
    test_data = RankDataSet(test_data, FLAGS.use_preds)
    
    valid_data = copy(dataset)
    valid_data.ranges = valid_ranges
    valid_data = RankDataSet(valid_data, FLAGS.use_preds)

    train_dl = DataLoader(train_data, shuffle=True)
    valid_dl = DataLoader(valid_data, shuffle=True)
    test_dl = DataLoader(test_data, shuffle=True)

    print("Training ndcg:", eval_random(train_dl))
    print("Validation ndcg:", eval_random(valid_dl))
    print("Test ndcg:", eval_random(test_dl))

def eval_random(dl):
    ndcgs = []
    for _, labels in dl:
        labels = labels.float().reshape(-1, 1)
        print(labels)
        ndcg = evaluate_ndcg_at_k(labels.cpu(), np.random.rand(len(labels)), 0)

        ndcgs.append(ndcg)
    return np.mean(ndcgs)

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = "Data/",
                        help='Directory for storing input data')
    parser.add_argument('--filename', type=str, default="dataset_filename=MIMICS-ClickExplore.tsv_expanded=True_balance=False_impression=False_reduced_classes=False_embedder=Bert_negative_samples=True.p",
                        help='Filename of the data')
    parser.add_argument('--use_preds', type=int, default=0,
                        help='Use the predictions in the ranker')
    FLAGS, unparsed = parser.parse_known_args()
    test()