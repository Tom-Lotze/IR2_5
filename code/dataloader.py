# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-09-15 01:35
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-09-21 18:28

import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
import copy
import numpy as np
import pickle as pkl
from tqdm import tqdm
import argparse
import os
import itertools
from collections import Counter, defaultdict

FLAGS = None

class Data():
    """
    A class to save all the preprocessed data in
    """
    def __init__(self, queries, questions, answers, impression_lvls, 
                 engagement_lvls, click_probs, query_embeds, question_embeds, 
                 answer_embeds):
        """
        Initializes Data object with all things that will be saved.
        """
        self.queries = queries
        self.questions = questions
        self.answers = answers
        self.impression_lvls = impression_lvls
        self.engagement_lvls = engagement_lvls
        self.click_probs = click_probs
        self.query_embeds = query_embeds
        self.question_embeds = query_embeds
        self.answer_embeds = answer_embeds
        self.ranges = self.get_ranges(self.queries)

    def get_ranges(self, queries):
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

def load():
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

    filename_dataset = f"dataset_filename={FLAGS.filename}_expanded={FLAGS.expanded}_balance={FLAGS.balance}_impression={FLAGS.impression}_bins={FLAGS.bins}_embedder={FLAGS.embedder}.p"

    # Check if loadable file exists
    if not os.path.exists(FLAGS.folder):
        raise OSError(f"Folder {FLAGS.folder} does not exist")
    if not os.path.exists(FLAGS.folder+FLAGS.filename):
        raise OSError(f"File {FLAGS.folder+FLAGS.filename} does not exist")

    with open(FLAGS.folder+FLAGS.filename) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)

        for line in tsvreader:
            if FLAGS.impression and line[7] == "low":
                continue

            queries.append(line[0])
            questions.append(line[1])
            answers.append([line[i] for i in range(2, 7)])
            impression_lvls.append(line[7])
            engagement_lvls.append(int(line[8]))
            click_probs.append([float(line[i]) for i in range(9, 14)])

    if FLAGS.balance:
        engagement_lvls = np.array(engagement_lvls)
        zero_indices = np.where(engagement_lvls == 0)[0]
        non_zero_indices = np.where(engagement_lvls != 0)[0]
        median_size = int(np.median(list(Counter(engagement_lvls).values())))
        sampled_indices = np.random.choice(zero_indices, median_size, replace=False)
        indices = np.concatenate((sampled_indices, non_zero_indices))

        queries = [queries[i] for i in indices]
        questions = [questions[i] for i in indices]
        answers = [answers[i] for i in indices]
        impression_lvls = [impression_lvls[i] for i in indices]
        engagement_lvls = [engagement_lvls[i] for i in indices]
        click_probs = [click_probs[i] for i in indices]

        answers = [i for sublist in answers for i in sublist]
        click_probs = [i for sublist in click_probs for i in sublist]

    # set language model
    if FLAGS.embedder == "Bert":
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

        question_embeds = torch.from_numpy(question_embeds)
        query_embeds = torch.from_numpy(query_embeds)
        answer_embeds = torch.from_numpy(answer_embeds)

    elif FLAGS.embedder == "TFIDF":
        #TODO
        raise NotImplementedError()
    else:
        print(f"Embedder {FLAGS.embedder} does not exist")
        return

    if  not FLAGS.expanded:
        dataset = []

        for i, (query, question) in tqdm(enumerate(zip(query_embeds, question_embeds))):

            # reshape answers
            answers = answer_embeds[i*5:i*5+5]
            answers = answers.reshape(-1)

            engagement_lvl = torch.Tensor([int(engagement_lvls[i])]).float()

            inp = torch.cat((query, question, answers), 0)

            # Add the datapoint to the dataset
            dataset.append((inp, engagement_lvl))

        # save the dataloader final time
        with open(filename_dataset, "wb") as f:
            pkl.dump(dataset, f)
    else:
        answers = list(zip(*[iter(answers)]*5))
        click_probs = list(zip(*[iter(click_probs)]*5))
        answer_embeds = answer_embeds.reshape(-1, 5)

        dataset = Data(queries, questions, answers, impression_lvls, 
                       engagement_lvls, click_probs, query_embeds, 
                       question_embeds, answer_embeds)

        # save the dataloader final time
        with open(filename_dataset, "wb") as f:
            pkl.dump(dataset, f, protocol=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expanded', type=bool, default=False, 
                        help='Return the old type of datastructure')
    parser.add_argument('--balance', type=bool, default=True, 
                        help='Balance the data by fixing the distributions')
    parser.add_argument('--folder', type=str, default='Data/', 
                        help='Folder where the data is located')
    parser.add_argument('--filename', type=str, default="MIMICS-Click.tsv",
                        help='Filename of the data')
    parser.add_argument('--impression', type=bool, default=True,
                        help='Use only the most shown clarification panes')
    parser.add_argument('--bins', type=int, default=11,
                        help='Number of classes to consider')
    parser.add_argument('--embedder', type=str, default="Bert",
                        help='Type of embedding use to represent sentence')
    
    FLAGS, unparsed = parser.parse_known_args()

    load()

