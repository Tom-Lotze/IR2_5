# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-09-18 11:21
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-09-18 11:42


import argparse
import numpy as np
import os
from regression import Regression
import torch
import matplotlib.pyplot as plt
import pickle as pkl

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '784, 392, 256, 128, 16'
LEARNING_RATE_DEFAULT = 1e-3
NR_EPOCHS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

FLAGS = None



def train():
    """
    Performs training and evaluation of Regression model.

    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Get number of units in each hidden layer
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []


    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)


    # initialize optimizer
    if FLAGS.optimizer == "SGD":
        optimizer = torch.optim.SGD(nn.parameters(), lr=FLAGS.learning_rate,
            weight_decay=FLAGS.weightdecay,
            momentum=FLAGS.momentum)
    # elif FLAGS.optimizer == "Adam":
    #   optimizer = torch.optim.Adam(nn.parameters(), lr=FLAGS.learning_rate,
    #     amsgrad=FLAGS.amsgrad, weight_decay=FLAGS.weightdecay)
    # elif FLAGS.optimizer == "AdamW":
    #   optimizer = torch.optim.AdamW(nn.parameters(), lr=FLAGS.learning_rate,
    #     amsgrad=FLAGS.amsgrad, weight_decay=FLAGS.weightdecay)
    # elif FLAGS.optimizer == "RMSprop":
    #   optimizer = torch.optim.RMSprop(nn.parameters(), lr=FLAGS.learning_rate,
    #     weight_decay=FLAGS.weightdecay, momentum=FLAGS.momentum)

    # initialization for plotting and metrics
    training_losses = []
    test_losses = []

    # extract training data
    dataloader = pkl.load("Data/dataloader.p")
     # initialize MLP
    nn = Regression(dataloader.batch_size, dnn_hidden_units, 1, FLAGS.neg_slope).to(device)
    crossEntropy = torch.nn.MSELoss()

    # perform the forward step, backward step and updating of weights max_steps number of times,
    for epoch in range(FLAGS.nr_epochs):

        print(f"Epoch: {epoch}")

        for batch, (x, y) in enumerate(dataloader):

            x = x.reshape(x.shape[0], -1).to(device)
            x = x.reshape(y.shape[0], -1).to(device)

            optimizer.zero_grad()

            pred = nn(x)

            # train_acc = accuracy(pred, y)

            # compute cross entropy loss
            loss = crossEntropy(pred, y)





            loss.backward(step)

            optimizer.step()


            # # evaluation on test set
            # if step % FLAGS.eval_freq == 0:
            #     test_accuracies, test_losses = eval_on_test(nn, crossEntropy, x_test, y_test, test_accuracies, test_losses)

            training_losses.append(loss.item())
            if batch % 100 == 0:
                print(f"Training loss batch {batch}: {loss.item()}")


    # # compute loss and accuracy on the test set a final time
    # test_accuracies, test_losses = eval_on_test(nn, crossEntropy, x_test, y_test, test_accuracies, test_losses)
    # print("Maximum accuracy :", max(test_accuracies), "after %d steps\n"%(np.argmax(test_accuracies) * FLAGS.eval_freq))

    # plotting(training_losses, test_losses, training_accuracies, test_accuracies, dnn_hidden_units, max(test_accuracies))

def eval_on_test(nn, crossEntropy, x_test, y_test, test_accuracies, test_losses):
    """
    Find the accuracy and loss on the test set, given the current weights
    """
    test_pred = nn(x_test)
    true_labels = torch.max(y_test, 1)[1]
    test_acc = accuracy(test_pred, y_test)
    test_loss = crossEntropy(test_pred, true_labels)
    print("Test accuracy is:", test_acc, "\n")
    test_accuracies.append(test_acc)
    test_losses.append(test_loss)

    return test_accuracies, test_losses


def plotting(train_losses, test_losses, train_accuracies, test_accuracies, dnn_hidden_units, max_test):
    plt.figure(figsize=(15, 12))
    steps_all = np.arange(1, FLAGS.max_steps+1)
    steps_test =  np.arange(0, FLAGS.max_steps+1, FLAGS.eval_freq)

    # plot the accuracies
    plt.subplot(2, 1, 1)
    plt.plot(steps_all, train_accuracies, '-', lw=2, label="Train accuracy")
    plt.plot(steps_test, test_accuracies, '-', lw=2, label="Test accuracy")

    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for training and test data')
    plt.grid(True)
    plt.legend()

    # plot the losses
    plt.subplot(2, 1, 2)
    plt.plot(steps_all, train_losses, '-', lw=2, label="Training loss")
    plt.plot(steps_test, test_losses, '-', lw=2, label="Test loss")
    plt.title('Loss for training and test data')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    dnn_string = "_".join([str(i) for i in dnn_hidden_units])

    fig_name = "max_accuracy="+str(max_test)+"_MLP_Torch_lr="+str(FLAGS.learning_rate)+ \
    "_neg_sl="+str(FLAGS.neg_slope)+ "_optimizer="+ FLAGS.optimizer+ \
    "_amsgrad="+str(int(FLAGS.amsgrad))+"_weightdecay=" + \
    str(FLAGS.weightdecay)+ "_momentum="+str(FLAGS.momentum)+"_dnn="+dnn_string+".png"

    plt.savefig("./images/"+"best"+fig_name)

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
    # parser.add_argument('--amsgrad', type=int, default=0,
    #                     help='Boolean: Amsgrad for Adam and Adamw')
    parser.add_argument('--weightdecay', type=float, default=0,
      help='weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0,
      help='momentum for optimizer')

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.amsgrad = bool(FLAGS.amsgrad)

    main()


