# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-09-15 01:35
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-09-18 15:48

import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
import copy
import numpy as np
import pickle as pkl
from tqdm import tqdm






def get_embedding(inp, model):
    """Find the embeddings for a string input, using a specified model"""
    # tokenize the raw input
    tokenized_input = tokenizer(inp, return_tensors="pt")

    # return pytorch tensors from the tokenized input
    embedding = model(**tokenized_input)[0].detach()

    return pooled_hidden_layer



data = []
questions = []
queries = []
answers = []
with open("Data/MIMICS-Click.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        # data.append(line)
        queries.append(line[0])
        questions.append(line[1])
        answers.extend([line[i] for i in range(2, 7)])

# headers = data[0]
# data = data[1:]
queries = queries[1:]
questions = questions[1:]



# set language model
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


question_embeds = embedder.encode(questions, convert_to_tensor=True, show_progress_bar=True, batch_size=128, num_workers = 4)

with open(f"Data/question_embeds_distilbert.p", "wb") as f:
            pkl.dump(question_embeds, f)

query_embeds = embedder.encode(queries, convert_to_tensor=True, show_progress_bar=True, batch_size=128, num_workers = 4)

with open(f"Data/query_embeds_distilbert.p", "wb") as f:
            pkl.dump(query_embeds, f)

answer_embeds = embedder.encode(answers, convert_to_tensor=True, show_progress_bar=True, batch_size=256, num_workers = 4)

with open(f"Data/answers_embeds_distilbert.p", "wb") as f:
            pkl.dump(answer_embeds, f)


dataset = []

for i, (query, question) in tqdm(enumerate(zip(query_embeds, question_embeds))):

    if i % 500 == 0:
        print(i)

    answers = answer_embeds[i:i+5]


    # convert to right types
    # impression_lvl = {"low": 1, "medium": 2, "high": 3}[impression_lvl]
    engagement_lvl = torch.Tensor([int(engagement_lvl)]).float()
    # ccp1, ccp2, ccp3, cpp4, ccp5 = float(ccp1), float(ccp2), float(ccp3), float(cpp4), float(ccp5)


    q = torch.cat((query, question), 1)

    # nr_options = len([op for op in [op1, op2, op3, op4, op5] if op])
    # batch_tensor = (query, question, nr_options, impression_lvl)

    # Add the datapoint to the dataset
    dataset.append((q, engagement_lvl))


# convert to pytorch dataloader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# save the dataloader final time
with open("Data/dataloader_query_question_pooled.p", "wb") as f:
    pkl.dump(dataloader, f)