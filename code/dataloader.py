# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-09-15 01:35
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-10-12 15:02

import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
import numpy as np
import pickle as pkl
from tqdm import tqdm
import argparse
import os
import itertools
from regression import Regression
from collections import Counter, defaultdict

FLAGS = None

class Data():
    """
    A class to save all the preprocessed data in
    """
    def __init__(self, queries, questions, answers, impression_lvls,
                 engagement_lvls, click_probs, predictions):
        """
        Initializes Data object with all things that will be saved.
        """
        self.queries = queries
        self.questions = questions
        self.answers = answers
        self.impression_lvls = impression_lvls
        self.engagement_lvls = engagement_lvls
        self.click_probs = click_probs
        self.predictions = predictions

        self.ranges = get_ranges(self.queries)

def get_ranges(queries):
    """
    Get ranges of what queries should be together.

    Args:
        queries: the queries in string form
    Returns:
        out: a list of list with all indices of the same query in a sublist
    """
    indices = defaultdict(list)
    for i, q in enumerate(queries):
        indices[q].append(i)
    return list(indices.values())

def load(FLAGS):
    """
    Load all data and store it in either a list (old) or in a dataset class (new)
    """
    questions = []
    queries = []
    answers = []
    impression_lvls = []
    engagement_lvls = []
    click_probs = []

    np.random.seed(42)

    filename_dataset = f"Data/dataset_filename={FLAGS.filename}_expanded={FLAGS.expanded}_balance={FLAGS.balance}_impression={FLAGS.impression}_reduced_classes={FLAGS.reduced_classes}_embedder={FLAGS.embedder}_negative_samples={FLAGS.negative_samples}.p"

    # Check if loadable file exists
    if not os.path.exists(FLAGS.folder):
        raise OSError(f"Folder {FLAGS.folder} does not exist")
    if not os.path.exists(FLAGS.folder+FLAGS.filename):
        raise OSError(f"File {FLAGS.folder+FLAGS.filename} does not exist")

    N = 500

    with open(FLAGS.folder+FLAGS.filename) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        # skip the first line (consists of labels)
        next(tsvreader, None)

        for i, line in enumerate(tsvreader):
            # skip the instances that have a low impression level
            if FLAGS.impression and line[7] == "low":
                continue

            # if i == N:
            #     break

            # Add values to the data lists
            queries.append(line[0])
            questions.append(line[1])
            answers.append([line[i] for i in range(2, 7)])
            impression_lvls.append(line[7])
            if FLAGS.reduced_classes:
                engagement_lvls.append(0 if int(line[8]) == 0 else 1)
            else:
                engagement_lvls.append(int(line[8]))
            click_probs.append([float(line[i]) for i in range(9, 14)])

    # Attempt to fix class imbalance assuming 0 is to large
    if FLAGS.balance:
        # Index the locations of zeros and non-zeros
        engagement_lvls = np.array(engagement_lvls)
        zero_indices = np.where(engagement_lvls == 0)[0]
        non_zero_indices = np.where(engagement_lvls != 0)[0]

        # Get the median size of the engagement levels
        if FLAGS.reduced_classes:
            median_size = int(Counter(engagement_lvls)[1])
        else:
            median_size = int(np.median(list(Counter(engagement_lvls).values())))

        # Return the to be used indices
        sampled_indices = np.random.choice(zero_indices, median_size, replace=False)
        indices = np.concatenate((sampled_indices, non_zero_indices))

        # Update datalist based on indices
        queries = [queries[i] for i in indices]
        questions = [questions[i] for i in indices]
        answers = [answers[i] for i in indices]
        impression_lvls = [impression_lvls[i] for i in indices]
        engagement_lvls = [engagement_lvls[i] for i in indices]
        click_probs = [click_probs[i] for i in indices]
    
    # TODO SAMPLE HERE

    if FLAGS.expanded and FLAGS.negative_samples:
        n_questions = len(questions)
        ranges = get_ranges(queries)
        sampled_question_indices = []

        for r in ranges:
            samples = np.random.choice([i for i in range(n_questions) if i not in r], FLAGS.sample_size, replace=False)
            sampled_question_indices.append(samples)
            max_engagement = np.max([engagement_lvls[i] for i in r])
            for i in r:
                if engagement_lvls[i] == max_engagement:
                    engagement_lvls[i] = 2
                else:
                    engagement_lvls[i] = 1


    # set language model
    if FLAGS.embedder == "Bert":
        # Flatten to load into embedder
        answers = [i for sublist in answers for i in sublist]

        embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

        question_embeds = embedder.encode(questions, convert_to_tensor=False,
                                        show_progress_bar=True, batch_size=128,
                                        num_workers = 4)

        query_embeds = embedder.encode(queries, convert_to_tensor=False,
                                    show_progress_bar=True, batch_size=128,
                                    num_workers = 4)

        answer_embeds = embedder.encode(answers, convert_to_tensor=False,
                                        show_progress_bar=True, batch_size=128,
                                        num_workers = 4)

        query_embeds = torch.from_numpy(query_embeds)
        question_embeds = torch.from_numpy(question_embeds)
        answer_embeds = torch.from_numpy(answer_embeds)

        print(query_embeds.shape)
        print(question_embeds.shape)
        print(answer_embeds.shape)

        answers = list(zip(*[iter(answers)]*5))

        if FLAGS.expanded and FLAGS.negative_samples:
            answer_embeds = list(answer_embeds.reshape(query_embeds.shape[0], -1))
            question_embeds = list(question_embeds)
            query_embeds = list(query_embeds)
            

            for r, samples in zip(ranges, sampled_question_indices):
                queries.extend([queries[r[0]]] * len(samples))
                questions.extend([questions[i] for i in samples])
                answers.extend([answers[i] for i in samples])
                impression_lvls.extend([impression_lvls[i] for i in samples])
                engagement_lvls.extend([0] * len(samples))
                click_probs.extend([click_probs[i] for i in samples])
                query_embeds.extend([query_embeds[r[0]]] * len(samples))
                question_embeds.extend([question_embeds[i] for i in samples])
                answer_embeds.extend([answer_embeds[i] for i in samples])

            query_embeds = torch.stack(query_embeds)
            question_embeds = torch.stack(question_embeds)
            answer_embeds = torch.stack(answer_embeds)
            print(query_embeds.shape)
            print(question_embeds.shape)
            print(answer_embeds.shape)

    elif FLAGS.embedder == "TFIDF":
        # initialize the vectorized
        if FLAGS.expanded:
            with open(f"{FLAGS.folder}TFIDF_vocab.p") as f:
                vocab = pkl.load(f)
            vectorizer = TfidfVectorizer(vocabulary=vocab)
        else:
            vectorizer = TfidfVectorizer()

        if FLAGS.expanded and FLAGS.negative_samples:
            for r, samples in zip(ranges, sampled_question_indices):
                queries.extend([queries[r[0]]] * len(samples))
                questions.extend([questions[i] for i in samples])
                answers.extend([answers[i] for i in samples])
                impression_lvls.extend([impression_lvls[i] for i in samples])
                engagement_lvls.extend([0] * len(samples))
                click_probs.extend([click_probs[i] for i in samples])

        # create the corpus: a list of string, each string is a data instance
        corpus = [" ".join([queries[i], questions[i], " ".join(answers[i])]) for i in range(len(queries))]

        # this yields a sparse vector
        X = vectorizer.fit_transform(corpus)
        if not FLAGS.expanded:
            with open(f"{FLAGS.folder}TFIDF_vocab.p", "wb") as f:
                pkl.dump(vectorizer.vocabulary_, f)

        # use code snippet from https://ray075hl.github.io/ray075hl.github.io/sparse_matrix_pytorch/ to convert to torch tensor
        X = X.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((X.row, X.col))).long()
        values = torch.from_numpy(X.data)
        shape = torch.Size(X.shape)
        X = torch.sparse_coo_tensor(indices, values, shape)
        print(f"shape of X: {X.shape}")

    else:
        print(f"Embedder {FLAGS.embedder} does not exist")
        return

    # either return the dataset for regression, with only questios, queries
    # and answers, or return with all attributes
    if FLAGS.expanded:
        # TODO
        # if statement if TFIDF or BERT
        # load neural net and perform forward pass on the data, yielding the predicted engagement levels
        if FLAGS.embedder == "Bert":
            answer_embeds = answer_embeds.reshape(query_embeds.shape[0], -1)
            
            input_matrix = torch.cat((query_embeds, question_embeds, answer_embeds), dim=1)

            nn = Regression(n_inputs=input_matrix.shape[1], 
                            n_hidden=[300,32], dropout_percentages=[0.0,0.0], 
                            n_classes=1, 
                            batchnorm=True)
            nn.load_state_dict(torch.load("Models/Best_regression_model.pt"))
            nn.eval()
            with torch.no_grad():
                preds = nn(input_matrix).squeeze()
        elif FLAGS.embedder == "TFIDF":
            nn = Regression(n_inputs=X.shape[1], 
                            n_hidden=[300,32], dropout_percentages=[0.0,0.0], 
                            n_classes=1, 
                            batchnorm=True)
            # TODO Correct model
            nn.load_state_dict(torch.load("Models/Regression_Bert_SGD_0.0001_1e-05_300, 32_0.0, 0.0_True_40.pt"))
            nn.eval()
            with torch.no_grad():
                preds = nn(X).squeeze()
        
        # Save in Data object
        dataset = Data(queries, questions, answers, impression_lvls,
                       engagement_lvls, click_probs, preds)

        # save the dataloader
        with open(filename_dataset, "wb") as f:
            pkl.dump(dataset, f, protocol=4)
    # return the dataset for regression
    else:
        dataset = []
        if FLAGS.embedder == "Bert":
            for i, (query, question) in tqdm(enumerate(zip(query_embeds, question_embeds))):

                # reshape answers
                answers = answer_embeds[i*5:i*5+5]
                answers = answers.reshape(-1)

                engagement_lvl = torch.Tensor([int(engagement_lvls[i])]).float()

                inp = torch.cat((query, question, answers), 0)

                # Add the datapoint to the dataset
                dataset.append((inp, engagement_lvl))
        
        elif FLAGS.embedder == "TFIDF":
            for i, inp in enumerate(X):
                dataset.append((inp, torch.Tensor([int(engagement_lvls[i])]).float()))

        # save the dataloader
        with open(filename_dataset, "wb") as f:
            pkl.dump(dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expanded', type=int, default=0,
                        help='Return the old type of datastructure')
    parser.add_argument('--balance', type=int, default=0,
                        help='Balance the data by fixing the distributions')
    parser.add_argument('--folder', type=str, default='Data/',
                        help='Folder where the data is located')
    parser.add_argument('--filename', type=str, default="MIMICS-ClickExplore.tsv",
                        help='Filename of the data')
    parser.add_argument('--impression', type=int, default=0,
                        help='Use only the most shown clarification panes')
    parser.add_argument('--reduced_classes', type=int, default=0,
                        help='Either consider 11 classes, or 2 (binary case)')
    parser.add_argument('--embedder', type=str, default="Bert",
                        help='Type of embedding use to represent sentence, either Bert or TFIDF')
    parser.add_argument('--negative_samples', type=int, default=1,
                        help='Use negative sampling to get a better reading out of the ranker')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='Number of negative samples to use')

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.expanded = bool(FLAGS.expanded)
    FLAGS.balance = bool(FLAGS.balance)
    FLAGS.impression = bool(FLAGS.impression)
    FLAGS.reduced_classes = bool(FLAGS.reduced_classes)
    FLAGS.negative_samples = bool(FLAGS.negative_samples)

    load(FLAGS)

