# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-09-15 01:35
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-09-21 15:56

import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
import copy
import numpy as np
import pickle as pkl
from tqdm import tqdm




data = []
questions = []
queries = []
answers = []
engagement_lvls = []

with open("Data/MIMICS-Click.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        queries.append(line[0])
        questions.append(line[1])
        answers.extend([line[i] for i in range(2, 7)])
        engagement_lvls.append(line[8])


# remove headers
queries = queries[1:]
questions = questions[1:]
answers = answers[1:]
engagement_lvls = engagement_lvls[1:]



# set language model
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


question_embeds = embedder.encode(questions, convert_to_tensor=False, show_progress_bar=True, batch_size=128, num_workers = 4)
# with open(f"Data/question_embeds_distilbert.p", "wb") as f:
#             pkl.dump(question_embeds, f)

query_embeds = embedder.encode(queries, convert_to_tensor=False, show_progress_bar=True, batch_size=128, num_workers = 4)
# with open(f"Data/query_embeds_distilbert.p", "wb") as f:
#             pkl.dump(query_embeds, f)

answer_embeds = embedder.encode(answers, convert_to_tensor=False, show_progress_bar=True, batch_size=128, num_workers = 4)
# with open(f"Data/answers_embeds_distilbert.p", "wb") as f:
#             pkl.dump(answer_embeds, f)

question_embeds = torch.from_numpy(question_embeds)
query_embeds = torch.from_numpy(query_embeds)
answer_embeds = torch.from_numpy(answer_embeds)


dataset = []

for i, (query, question) in tqdm(enumerate(zip(query_embeds, question_embeds))):

    answers = answer_embeds[i*5:i*5+5]

    # reshape answers
    answers = answers.reshape(-1)

    # convert to right types
    # impression_lvl = {"low": 1, "medium": 2, "high": 3}[impression_lvl]
    engagement_lvl = torch.Tensor(int(engagement_lvls[i])).float()
    # ccp1, ccp2, ccp3, cpp4, ccp5 = float(ccp1), float(ccp2), float(ccp3), float(cpp4), float(ccp5)


    inp = torch.cat((query, question, answers), 0)

    # nr_options = len([op for op in [op1, op2, op3, op4, op5] if op])
    # batch_tensor = (query, question, nr_options, impression_lvl)

    # Add the datapoint to the dataset
    dataset.append((inp, engagement_lvl))


# convert to pytorch dataloader
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)



# save the dataloader final time
with open("Data/dataset.p", "wb") as f:
    pkl.dump(dataset, f)
