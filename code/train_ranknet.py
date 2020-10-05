import torch
from datasets import RankDataSet
import argparse
import numpy as np
import os
import pickle as pkl
from ranknet import RankNet
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '256, 128, 32'
DROPOUT_DEFAULT = '0, 0, 0'
LEARNING_RATE_DEFAULT = 1e-3
NR_EPOCHS_DEFAULT = 500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02
DATA_DIR_DEFAULT = "dataloader/"


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    # if not os.path.exists(FLAGS.data_dir):
    # os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()

def train():
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Get number of units in each hidden layer
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # convert dropout percentages
    dropout_percentages = [int(perc) for perc in FLAGS.dropout_percentages.split(',')]

    # check if length of dropout is equal to nr of hidden layers
    if len(dropout_percentages) != len(dnn_hidden_units):
        dropout_len = len(dropout_percentages)
        hidden_len = len(dnn_hidden_units)
        if dropout_len < hidden_len:
            for _ in range(hidden_len-dropout_len):
                dropout_percentages.append(0)
        else:
            dropout_percentages = dropout_percentages[:hidden_len]
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    # extract all data and divide into train, valid and split dataloaders
    with open(os.path.join(FLAGS.data_dir, FLAGS.filename), "rb") as f:
        dataset = pkl.load(f)
      
    #TODO datasplits?

    

    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_dl = DataLoader(valid_data, batch_size=64, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)
    
    # initialize MLP
    nn = RankNet(5, dnn_hidden_units, dropout_percentages, 1, FLAGS.neg_slope, FLAGS.batchnorm).to(device)

    # initialize optimizer
    if FLAGS.optimizer == "SGD":
        optimizer = torch.optim.SGD(nn.parameters(), lr=FLAGS.learning_rate,
                                    weight_decay=FLAGS.weightdecay,
                                    momentum=FLAGS.momentum)
    elif FLAGS.optimizer == "Adam":
        optimizer = torch.optim.Adam(nn.parameters(), lr=FLAGS.learning_rate,
                                     amsgrad=FLAGS.amsgrad, 
                                     weight_decay=FLAGS.weightdecay)
    elif FLAGS.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(nn.parameters(), lr=FLAGS.learning_rate,
                                      amsgrad=FLAGS.amsgrad, 
                                      weight_decay=FLAGS.weightdecay)
    elif FLAGS.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(nn.parameters(), lr=FLAGS.learning_rate,
                                        weight_decay=FLAGS.weightdecay, 
                                        momentum=FLAGS.momentum)

    
    for epoch in tqdm(range(FLAGS.nr_epochs)):
         for step, (x_batch, y_batch) in enumerate(train_dl):
                
                optimizer.zero_grad()

                num_docs = x_batch.shape[1]
                num_pairs = num_docs * num_docs

                # ignore batch if only one doc (no pairs)
                if num_docs == 1:
                    continue

                # squeeze batch
                x_batch = x_batch.float().squeeze()
                y_batch = y_batch.float().t()

                # construct labels matrix
                labels_mat = y_batch.t() - y_batch
                labels_mat[labels_mat > 0] = 1
                labels_mat[labels_mat == 0] = 0
                labels_mat[labels_mat < 0] = -1

                # perform forward pass and compute lambdas
                scores = nn(x_batch)
                diff_mat = torch.sigmoid(torch.add(scores.t(), -scores))

                lambda_ij = (1/2) * (1 - labels_mat) - diff_mat
                lambda_i = lambda_ij.sum(dim=0)

                # perform backward pass and correct for number of pairs
                scores.squeeze().backward(lambda_i / num_pairs)
                optimizer.step()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--dropout_percentages', type = str, default = DROPOUT_DEFAULT,
      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
      help='Learning rate')
    parser.add_argument('--nr_epochs', type = int, default = NR_EPOCHS_DEFAULT,
      help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
      help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
    help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
      help='Directory for storing input data')
    parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
      help='Negative slope parameter for LeakyReLU')
    parser.add_argument('--optimizer', type=str, default="SGD",
      help='Type of optimizer')
    parser.add_argument('--amsgrad', type=int, default=0,
                        help='Boolean: Amsgrad for Adam and Adamw')
    parser.add_argument('--batchnorm', type=int, default=0,
                        help='Boolean: apply batch normalization?')
    parser.add_argument('--weightdecay', type=float, default=0,
      help='weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0,
      help='momentum for optimizer')
    parser.add_argument('--filename', type=str, default="MIMICS-ClickExplore.p",
                        help='Filename of the data')

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.amsgrad = bool(FLAGS.amsgrad)
    FLAGS.batchnorm = bool(FLAGS.batchnorm)

    main()