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
    questions = []
    queries = []
    answers = []
    impression_lvls = []
    engagement_lvls = []
    click_probs = []

    if not os.path.exists(FLAGS.folder):
        raise OSError(f"Folder {FLAGS.folder} does not exist")
    if not os.path.exists(FLAGS.folder+FLAGS.filename):
        raise OSError(f"File {FLAGS.folder+FLAGS.filename} does not exist")

    with open(FLAGS.folder+FLAGS.filename) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)
        for line in tsvreader:
            queries.append(line[0])
            questions.append(line[1])
            answers.extend([line[i] for i in range(2, 7)])
            impression_lvls.append(line[7])
            engagement_lvls.append(line[8])
            click_probs.extend([line[i] for i in range(9, 14)])

    # set language model
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

    if FLAGS.old:
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
        with open("Data/dataset.p", "wb") as f:
            pkl.dump(dataset, f)
    else:
        answers = list(zip(*[iter(answers)]*5))
        click_probs = list(zip(*[iter(click_probs)]*5))
        answer_embeds = answer_embeds.reshape(-1, 5)

        dataset = Data(queries, questions, answers, impression_lvls, 
                       engagement_lvls, click_probs, query_embeds, 
                       question_embeds, answer_embeds)

        # save the dataloader final time
        with open((FLAGS.folder+FLAGS.filename).replace('tsv', 'p'), "wb") as f:
            pkl.dump(dataset, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=bool, default=False, 
                        help='Return the old type of datastructure')
    parser.add_argument('--folder', type=str, default='Data/', 
                        help='Folder where the data is located')
    parser.add_argument('--filename', type=str, default="MIMICS-Click.tsv",
                        help='Filename of the data')
    
    FLAGS, unparsed = parser.parse_known_args()

    load()

