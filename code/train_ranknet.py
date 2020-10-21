import torch
from datasets import RankDataSet
import argparse
import numpy as np
import os
import pickle as pkl
from ranknet import RankNet
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from copy import copy
from dataloader import Data
from matplotlib import pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '32, 16'
DROPOUT_DEFAULT = '0, 0, 0'
LEARNING_RATE_DEFAULT = 1e-3
NR_EPOCHS_DEFAULT = 1
EVAL_FREQ_DEFAULT = 5000
NEG_SLOPE_DEFAULT = 0.02
DATA_DIR_DEFAULT = "Data/"

def dcg_at_k(sorted_labels, k):
    """
    Get the dcg at k, if k = 0 get the whole dcg.
    """
    if k > 0:
        k = min(sorted_labels.shape[0], k)
    else:
        k = sorted_labels.shape[0]
    denom = 1./np.log2(np.arange(k)+2.)
    nom = 2**sorted_labels-1.
    dcg = np.sum(nom[:k]*denom)
    return dcg

def ndcg_at_k(sorted_labels, ideal_labels, k):
    """
    Get the ndcg at k, if k = 0 get the whole ndcg.
    """
    return dcg_at_k(sorted_labels, k) / dcg_at_k(ideal_labels, k)


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
    dropout_percentages = [int(perc) for perc in FLAGS.dropout_probs.split(',')]

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


    if FLAGS.use_preds:
      input_size = 5
    else:
      input_size = 4

    variables_string = f"ranking_{FLAGS.optimizer}_{FLAGS.learning_rate}_{FLAGS.weightdecay}_{FLAGS.momentum}_{FLAGS.dnn_hidden_units}_{FLAGS.dropout_probs}_{FLAGS.batchnorm}_{FLAGS.nr_epochs}_{FLAGS.use_preds}"

    # initialize MLP
    nn = RankNet(input_size, dnn_hidden_units, dropout_percentages, 1, FLAGS.neg_slope, FLAGS.batchnorm).to(device)

    # initialization for plotting and metrics
    training_ndcgs = []
    valid_ndcgs = []
    first_rel_ranks = []

    initial_train_ndcg, _ = eval_on_test(nn, train_dl, device)
    training_ndcgs.append(initial_train_ndcg)
    initial_valid_ndcg, _ = eval_on_test(nn, valid_dl, device)
    valid_ndcgs.append(initial_valid_ndcg)

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

    overall_batch = 0
    training_ndcg = []

    for epoch in range(FLAGS.nr_epochs):

        print(f"\nEpoch: {epoch}")

        for x_batch, y_batch in train_dl:

            optimizer.zero_grad()

            num_docs = x_batch.shape[1]
            num_pairs = num_docs * num_docs

            # ignore batch if only one doc (no pairs)
            if num_docs == 1:
                print("Query with only a single question found")
                continue

            # squeeze batch
            x_batch = x_batch.float().squeeze().to(device)
            y_batch = y_batch.float().reshape(-1, 1).to(device)

            # construct labels matrix
            labels_mat = y_batch.T - y_batch
            labels_mat[labels_mat > 0] = 1
            labels_mat[labels_mat == 0] = 0
            labels_mat[labels_mat < 0] = -1



            # perform forward pass and compute lambdas
            scores = nn(x_batch).to(device)
            # print("Batch:", overall_batch)
            # print("Max:", np.max(scores.detach().cpu().flatten().numpy()), "Min:", np.min(scores.detach().cpu().flatten().numpy()))
            # print((np.isnan(scores.detach().cpu().flatten().numpy()).any()))
            if np.isnan(scores.detach().cpu().flatten().numpy()).any():
              print([param for param in nn.parameters()])
              print("label mat", labels_mat)
              print("scores prev", scores)
              return

            diff_mat = torch.sigmoid(torch.add(scores.T, -scores))

            lambda_ij = (1/2) * (1 - labels_mat) - diff_mat
            lambda_i = lambda_ij.sum(dim=0)

            torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=10.0)

            # perform backward pass and correct for number of pairs
            scores.squeeze().backward(lambda_i / num_pairs)
            optimizer.step()

            labels, scores = np.array(y_batch.detach().cpu()).flatten(), np.array(scores.detach().cpu()).flatten()
            random_i = np.random.permutation(np.arange(scores.shape[0]))
            labels = labels[random_i]
            scores = scores[random_i]

            sort_ind = np.argsort(scores)[::-1]
            sorted_labels = labels[sort_ind]
            ideal_labels = np.sort(labels)[::-1]
            first_rel_rank = np.argmax(sorted_labels)
            # print(f"first rel rank: {first_rel_rank}")
            first_rel_ranks.append(first_rel_rank)
            ideal_labels = np.sort(labels)[::-1]

            training_ndcg.append(ndcg_at_k(sorted_labels, ideal_labels, 0))

            if overall_batch % FLAGS.eval_freq == 0 and overall_batch != 0:
                valid_ndcg, valid_MRR = eval_on_test(nn, valid_dl, device)
                valid_ndcgs.append(valid_ndcg)
                mean_training_ndcg = np.mean(training_ndcg)
                training_ndcgs.append(mean_training_ndcg)
                training_ndcg = []
                print(f"Training ndcg: {mean_training_ndcg} / Valid ndcg: {valid_ndcg}")
                print(f"average MRR: {np.mean(first_rel_ranks)}")

                #TODO Optimal model?

            overall_batch += 1

    torch.save(nn.state_dict(), f"Models/Ranker_{variables_string}.pt")

    test_loss, MRR_test = eval_on_test(nn, test_dl, device)

    print(f"training MRR: {np.mean(first_rel_ranks)}")
    print(f"Test Loss: {test_loss}, MRR test: {MRR_test}")

    if FLAGS.plotting:
        optimal_batch = 0
        plotting(training_ndcgs, valid_ndcgs, test_loss, variables_string, optimal_batch, FLAGS)

def plotting(train_losses, valid_losses, test_loss, variables_string, optimal_batch, FLAGS):
    plt.rcParams.update({"font.size": 22})

    print(train_losses)
    print(valid_losses)

    os.makedirs("Images", exist_ok=True)

    plt.figure(figsize=(15, 9))
    train_steps = np.arange(0, len(train_losses)) * FLAGS.eval_freq
    valid_steps = np.arange(0, len(valid_losses)) * FLAGS.eval_freq

    # plot the losses
    plt.plot(train_steps, train_losses, '-', lw=2, label="Training NDCG")
    plt.plot(valid_steps, valid_losses, '-', lw=3, label="Validation NDCG")
    plt.axhline(test_loss, label="Test NDCG", color="red", lw=3)
    # plt.axvline(optimal_batch, linestyle="dashed", color='red', label="Optimal model", lw=3)
    # plt.title('Losses over training, including final test loss using optimal model')

    # plt.ylim((0.46, 0.52))
    plt.xlabel('Batch')
    plt.ylabel('NDCG')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    fig_name = f"NDCGplot_{variables_string}.png"
    plt.savefig(f"Images/{fig_name}")

def eval_on_test(nn, dl, device):
    """
    Find the ndcg on the test set, given the current weights
    """
    nn.eval()
    nn = nn.to(device)

    with torch.no_grad():
        ndcgs = []
        first_rel_ranks = []
        for x_batch, y_batch in dl:

            x_batch = x_batch.float().squeeze().to(device)
            y_batch = y_batch.float().reshape(-1, 1).to(device)

            scores = nn(x_batch).to(device)

            labels, scores = np.array(y_batch.cpu()).flatten(), np.array(scores.detach().cpu()).flatten()
            random_i = np.random.permutation(np.arange(scores.shape[0]))
            labels = labels[random_i]
            scores = scores[random_i]

            sort_ind = np.argsort(scores)[::-1]
            sorted_labels = labels[sort_ind]
            ideal_labels = np.sort(labels)[::-1]
            first_rel_rank = np.argmax(sorted_labels)
            first_rel_ranks.append(first_rel_rank)

            ndcg = ndcg_at_k(sorted_labels, ideal_labels, 0)

            ndcgs.append(ndcg)

    return np.mean(ndcgs), np.mean(first_rel_ranks)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--dropout_probs', type = str, default = DROPOUT_DEFAULT,
      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
      help='Learning rate')
    parser.add_argument('--nr_epochs', type = int, default = NR_EPOCHS_DEFAULT,
      help='Number of epochs to run trainer.')
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
    parser.add_argument('--weightdecay', type=float, default=0.001,
      help='weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0,
      help='momentum for optimizer')
    parser.add_argument('--filename', type=str, default="dataset_filename=MIMICS-ClickExplore.tsv_expanded=True_balance=False_impression=False_reduced_classes=False_embedder=Bert_negative_samples=True.p",
                        help='Filename of the data')
    parser.add_argument('--use_preds', type=int, default=0,
      help='Use the predictions in the ranker')
    parser.add_argument('--plotting', type=int, default=0,
      help="Save a plot of the NDCG over the generations")

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.amsgrad = bool(FLAGS.amsgrad)
    FLAGS.batchnorm = bool(FLAGS.batchnorm)
    FLAGS.use_preds = bool(FLAGS.use_preds)
    FLAGS.plotting = bool(FLAGS.plotting)

    main()