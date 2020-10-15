# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-09-18 11:21
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-10-15 21:31


import argparse
import numpy as np
import os
from regression import Regression
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pickle as pkl
from collections import Counter


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '300, 32'
DROPOUT_DEFAULT = '0.0, 0.0'
LEARNING_RATE_DEFAULT = 1e-3
NR_EPOCHS_DEFAULT = 30
BATCH_SIZE_DEFAULT = 64
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02
DATA_DIR_DEFAULT = "Data/"

FLAGS = None

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train():
    """
    Performs training and evaluation of Regression model.
    """
    # Set the random seeds for reproducibility
    np.random.seed(10)
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get number of units in each hidden layer
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # convert dropout percentages
    dropout_probs = [float(prob) for prob in FLAGS.dropout_probs.split(',')]

    # check if length of dropout is equal to nr of hidden layers
    if len(dropout_probs) != len(dnn_hidden_units):
        dropout_len = len(dropout_probs)
        hidden_len = len(dnn_hidden_units)
        if dropout_len < hidden_len:
            for _ in range(hidden_len-dropout_len):
                dropout_probs.append(0)
        else:
            dropout_probs = dropout_probs[:hidden_len]
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    # extract all data and divide into train, valid and split dataloaders
    dataset_filename = f"dataset_filename=MIMICS-Click.tsv_expanded=False_balance=True_impression={FLAGS.impression}_reduced_classes={FLAGS.reduced_classes}_embedder={FLAGS.embedder}.p"
    with open(os.path.join(FLAGS.data_dir, dataset_filename), "rb") as f:
        dataset = pkl.load(f)

    len_all = len(dataset)

    train_len, valid_len = int(0.7 * len_all), int(0.15 * len_all)
    test_len = len_all - train_len - valid_len
    splits = [train_len, valid_len, test_len]
    train_data, valid_data, test_data = random_split(dataset, splits)

    train_dl = DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_data, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)


     # initialize MLP and loss function
    input_size = iter(train_dl).next()[0].shape[1] # 5376 for BERT embeddings
    nn = Regression(input_size, dnn_hidden_units, dropout_probs, 1, FLAGS.neg_slope, FLAGS.batchnorm).to(device)
    loss_function = torch.nn.MSELoss()

    if FLAGS.verbose:
        print(f"neural net:\n {[param.data for param in nn.parameters()]}")


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
    valid_losses = []

    initial_train_loss = eval_on_test(nn, loss_function, train_dl, device)
    training_losses.append(initial_train_loss)
    initial_valid_loss = eval_on_test(nn, loss_function, valid_dl, device)
    valid_losses.append(initial_valid_loss)


    # construct name for saving models and figures
    variables_string = f"regression_{FLAGS.embedder}_{FLAGS.impression}_{FLAGS.reduced_classes}_{FLAGS.optimizer}_{FLAGS.learning_rate}_{FLAGS.weightdecay}_{FLAGS.momentum}_{FLAGS.dnn_hidden_units}_{FLAGS.dropout_probs}_{FLAGS.batchnorm}_{FLAGS.nr_epochs}"

    overall_batch = 0
    min_valid_loss = 10000

    # training loop
    for epoch in range(FLAGS.nr_epochs):

        print(f"\nEpoch: {epoch}")


        for batch, (x, y) in enumerate(train_dl):
            nn.train()

            # squeeze the input, and put on device
            x = x.to(device)
            y = y.to(device)


            optimizer.zero_grad()

            # forward pass
            pred = nn(x).to(device)

            # compute loss and backpropagate
            loss = loss_function(pred, y)
            loss.backward()

            # update the weights
            optimizer.step()

            # save training loss
            training_losses.append(loss.item())

            # print(f"batch loss ({batch}): {loss.item()}")

            # get loss on validation set and evaluate
            if overall_batch % FLAGS.eval_freq == 0 and overall_batch != 0:
                valid_loss = eval_on_test(nn, loss_function, valid_dl, device)
                valid_losses.append(valid_loss)
                print(f"Training loss: {loss.item()} / Valid loss: {valid_loss}")
                if valid_loss < min_valid_loss:
                    print(f"Model is saved in epoch {epoch}, overall batch: {overall_batch}")
                    torch.save(nn.state_dict(), f"Models/Regression_{variables_string}.pt")
                    min_valid_loss = valid_loss
                    optimal_batch = overall_batch

            overall_batch += 1

    # Load the optimal model (with loweest validation loss, and evaluate on test set)
    optimal_nn = Regression(input_size, dnn_hidden_units, dropout_probs, 1, FLAGS.neg_slope, FLAGS.batchnorm).to(device)
    optimal_nn.load_state_dict(torch.load(f"Models/Regression_{variables_string}.pt"))


    test_loss, test_pred, test_true = eval_on_test(optimal_nn, loss_function, test_dl, device, verbose=FLAGS.verbose, return_preds=True)

    print(f"Loss on test set of optimal model (batch {optimal_batch}): {test_loss}")

    significance_testing(test_pred, test_true, loss_function, FLAGS)

    if FLAGS.plotting:
        plotting(training_losses, valid_losses, test_loss, variables_string, optimal_batch, FLAGS)



def significance_testing(test_preds, test_labels, loss_fn, FLAGS):

    print("\nImpression:", FLAGS.impression)
    print("Reduced Classes:", FLAGS.reduced_classes)

    print("Engagement levels:", Counter(test_labels))
    print(f"Total number of engagement levels: {len(test_labels)}\n")

    test_labels = torch.tensor(test_labels)
    test_pred = torch.tensor(test_preds)
    mean_eng = torch.mean(test_labels)
    median_eng = torch.median(test_labels)
    mode_eng = torch.mode(test_labels)[0]

    print(f"mean, median, mode: {mean_eng}, {median_eng}, {mode_eng}")


    mean = torch.full_like(test_labels, mean_eng)
    median = torch.full_like(test_labels, median_eng)
    mode = torch.full_like(test_labels, mode_eng)

    MSE_MODEL = loss_fn(test_preds, test_labels)
    MSE_mean = loss_fn(mean, test_labels)
    MSE_median = loss_fn(median, test_labels)
    MSE_mode = loss_fn(mode, test_labels)

    t_mean, p_mean = ttest_rel(mean, test_preds)
    t_median, p_median = ttest_rel(median, test_preds)
    t_mode, p_mode = ttest_rel(mode, test_preds)

    print(f"MSE MODEL Loss: {MSE_MODEL}")
    print(f"MSE mean Loss: {MSE_mean}, p-value: {p_mean}")
    print(f"MSE median Loss: {MSE_median}, p-value: {p_median}")
    print(f"MSE mode Loss: {MSE_mode}, p-value: {p_mode}")



def eval_on_test(nn, loss_function, dl, device, verbose=False, return_preds=False):
    """
    Find the accuracy and loss on the test set, given the current weights
    """
    all_predictions = []
    all_labels = []

    nn.eval()
    if verbose:
        print(f"neural net:\n {[param.data for param in nn.parameters()]}")

    nn.to(device)
    with torch.no_grad():
        losses = []
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)

            test_pred = nn(x).to(device)

            loss = loss_function(test_pred, y)

            losses.append(loss.item())

            if return_preds:
                all_predictions.extend((test_pred).tolist())
                all_labels.extend(y.squeeze().tolist())

            if verbose and i == 0:
                print(test_pred)

    mean_losses = np.mean(losses)

    if not return_preds:
        return mean_losses

    return mean_losses, all_predictions, all_labels


def plotting(train_losses, valid_losses, test_loss, variables_string, optimal_batch, FLAGS):
    plt.rcParams.update({"font.size": 22})

    os.makedirs("Images", exist_ok=True)

    plt.figure(figsize=(15, 9))
    steps_all = np.arange(0, len(train_losses))
    steps_valid = np.arange(0, len(valid_losses)) * FLAGS.eval_freq

    # plot the losses
    plt.plot(steps_all, train_losses, '-', lw=2, label="Training loss")
    plt.plot(steps_valid, valid_losses, '-', lw=3, label="Validation loss")
    plt.axhline(test_loss, label="Test loss", color="red", lw=3)
    plt.axvline(optimal_batch, linestyle="dashed", color='red', label="Optimal model", lw=3)
    # plt.title('Losses over training, including final test loss using optimal model')


    plt.xlabel('Batch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    fig_name = f"lossplot_{variables_string}.png"
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
    print("Training regression with following parameters:")
    print_flags()

    # Run the training operation
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str,
        default=DNN_HIDDEN_UNITS_DEFAULT,
        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--dropout_probs', type=str, default= DROPOUT_DEFAULT,
      help='Comma separated list of dropout probabilities in each layer')
    parser.add_argument('--learning_rate', type=float,
        default=LEARNING_RATE_DEFAULT, help='Learning rate')
    parser.add_argument('--nr_epochs', type=int, default = NR_EPOCHS_DEFAULT,
      help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type=int, default = BATCH_SIZE_DEFAULT,
      help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
    help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
      help='Directory for storing input data')
    parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
      help='Negative slope parameter for LeakyReLU')
    parser.add_argument('--optimizer', type=str, default="Adam",
      help='Type of optimizer')
    parser.add_argument('--amsgrad', type=int, default=1,
                        help='Boolean: Amsgrad for Adam and Adamw')
    parser.add_argument('--batchnorm', type=int, default=1,
                        help='Boolean: apply batch normalization?')
    # 0.0001 seems optimal
    parser.add_argument('--weightdecay', type=float, default=0,
      help='weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0,
      help='momentum for optimizer')
    parser.add_argument('--embedder', type=str, default='Bert',
      help='which dataset to use: TFIDF or Bert')
    parser.add_argument('--verbose', type=int, default=0,
      help='print neural net and predictions')
    parser.add_argument('--reduced_classes', type=int, default=0,
      help='Use only 2 class dataset')
    parser.add_argument('--impression', type=int, default=1,
      help='If true, filter low impression instances out')
    parser.add_argument('--plotting', type=int, default=1,
      help='if true, plots are saved')




    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.amsgrad = bool(FLAGS.amsgrad)
    FLAGS.batchnorm = bool(FLAGS.batchnorm)
    FLAGS.verbose = bool(FLAGS.verbose)
    FLAGS.reduced_classes = bool(FLAGS.reduced_classes)
    FLAGS.impression = bool(FLAGS.impression)
    FLAGS.plotting = bool(FLAGS.plotting)

    main()


