# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-09-18 11:21
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-10-06 14:05


import argparse
import numpy as np
import os
from classification import Classification
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pickle as pkl

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '300, 32'
DROPOUT_DEFAULT = '0, 0, 0'
LEARNING_RATE_DEFAULT = 1e-3
NR_EPOCHS_DEFAULT = 500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02
DATA_DIR_DEFAULT = "dataloader/"

FLAGS = None

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_accuracy(pred, y):
    with torch.no_grad():
        B = pred.shape[0]
        max_index = pred.max(dim = 1)[1]

    return (max_index == y).sum().item() / B


def train():
    """
    Performs training and evaluation of Classification model.
    """
    print("Training started")
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
    with open(os.path.join(FLAGS.data_dir, "dataset.p"), "rb") as f:
        dataset = pkl.load(f)

    len_all = len(dataset)

    train_len, valid_len = int(0.7 * len_all), int(0.15 * len_all)
    test_len = len_all - train_len - valid_len
    splits = [train_len, valid_len, test_len]
    train_data, valid_data, test_data = random_split(dataset, splits)

    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_dl = DataLoader(valid_data, batch_size=64, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)


     # initialize MLP and loss function

    nn = Classification(5376, dnn_hidden_units, dropout_percentages, 11, FLAGS.neg_slope, FLAGS.batchnorm).to(device)
    loss_function = torch.nn.CrossEntropyLoss()


    # initialize optimizer
    if FLAGS.optimizer == "SGD":
        optimizer = torch.optim.SGD(nn.parameters(), lr=FLAGS.learning_rate,
            weight_decay=FLAGS.weightdecay,
            momentum=FLAGS.momentum)
    elif FLAGS.optimizer == "Adam":
      optimizer = torch.optim.Adam(nn.parameters(), lr=FLAGS.learning_rate,
        amsgrad=FLAGS.amsgrad, weight_decay=FLAGS.weightdecay)
    elif FLAGS.optimizer == "AdamW":
      optimizer = torch.optim.AdamW(nn.parameters(), lr=FLAGS.learning_rate,
        amsgrad=FLAGS.amsgrad, weight_decay=FLAGS.weightdecay)
    elif FLAGS.optimizer == "RMSprop":
      optimizer = torch.optim.RMSprop(nn.parameters(), lr=FLAGS.learning_rate,
        weight_decay=FLAGS.weightdecay, momentum=FLAGS.momentum)

    # initialization for plotting and metrics
    training_losses = []
    training_accs = []
    valid_losses = []
    valid_accs = []

    # construct name for saving models and figures
    variables_string = f"{FLAGS.optimizer}_{FLAGS.learning_rate}_{FLAGS.weightdecay}_{FLAGS.dnn_hidden_units}_{FLAGS.dropout_percentages}_{FLAGS.batchnorm}_{FLAGS.nr_epochs}"

    # training loop
    for epoch in range(FLAGS.nr_epochs):

        print(f"\nEpoch: {epoch}")
        batch_losses = []
        nn.train()

        for batch, (x, y) in enumerate(train_dl):

            # squeeze the input, and put on device
            x = x.reshape(x.shape[0], -1).to(device)
            y = y.reshape(y.shape[0], -1).long().squeeze().to(device)

            optimizer.zero_grad()

            # forward pass
            pred = nn(x).to(device)

            # compute loss and backpropagate
            loss = loss_function(pred, y)
            loss.backward()

            # update the weights
            optimizer.step()

            # save training loss
            batch_losses.append(loss.item())
            acc = get_accuracy(pred, y)
            batch_accs.append(acc)

            print("batch loss", loss.item())
            print(f"accuracy: {acc}")

        avg_epoch_acc = np.mean(batch_accs)
        avg_epoch_loss = np.mean(batch_losses)
        training_losses.append(avg_epoch_loss)
        training_accs.append(avg_epoch_acc)
        print(f"Average batch loss & accuracy (epoch {epoch}): {avg_epoch_loss}, {avg_epoch_acc} ({len(batch_losses)} batches).")

        # get loss on validation set and evaluate
        valid_loss, valid_acc = eval_on_test(nn, loss_function, valid_dl, device)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        torch.save(nn.state_dict(), f"Models/Classification_{variables_string}.pt")


    # compute loss and accuracy on the test set
    test_loss, test_acc = eval_on_test(nn, loss_function, test_dl, device)
    print(f"Loss & accuracy on test set: {test_loss}, {test_acc}")

    plotting(training_losses, training_accs, valid_losses, valid_accs, test_loss, test_acc, variables_string)



def eval_on_test(nn, loss_function, dl, device):
    """
    Find the accuracy and loss on the test set, given the current weights
    """
    nn.eval()
    nn.to(device)
    with torch.no_grad():
        losses = []
        for (x, y) in dl:
            x = x.to(device)
            y = y.long().squeeze().to(device)

            test_pred = nn(x).to(device)

            loss = loss_function(test_pred, y)
            acc = get_accuracy(test_pred, y)
            losses.append(loss.item())a
            accs.append(acc)

    return np.mean(losses), np.mean(accs)


def plotting(train_losses, train_accs, valid_losses, valid_accs, test_loss, test_acc, variables_string):
    plt.rcParams.update({"font.size": 22})

    os.makedirs("Images", exist_ok=True)

    plt.figure(figsize=(20, 12))
    steps_all = np.arange(1, len(train_losses)+1)

    # plot the losses
    plt.subplot(2, 1, 1)
    plt.plot(steps_all, train_losses, '-', lw=2, label="Training loss")
    plt.plot(steps_all, valid_losses, '-', lw=2, label="Validation loss")
    plt.hlines(test_loss, 1, max(steps_all), label="Test loss")
    plt.title('Losses over training')

    # plt.ylim(0, 10)

    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(steps_all, train_accs, '-', lw=2, label="Training accuracy")
    plt.plot(steps_all, valid_accs, '-', lw=2, label="Validation accuracy")
    plt.hlines(test_acc, 1, max(steps_all), label="Test accuracy")
    plt.title('Accuracy over training')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    fig_name = f"classification_loss_acc_plot_{variables_string}.png"
    plt.savefig(f"Images/{fig_name}")

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
    # 0.0001 seems optimal
    parser.add_argument('--weightdecay', type=float, default=0,
      help='weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0,
      help='momentum for optimizer')

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.amsgrad = bool(FLAGS.amsgrad)
    FLAGS.batchnorm = bool(FLAGS.batchnorm)

    main()


