# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-09-15 01:35
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-09-17 13:46

import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AlbertTokenizer, AlbertModel
import copy
import numpy as np
import pickle






def get_embedding(inp, model="Albert"):
    """Find the embeddings for a string input, using a specified model"""
    # decide which language model to use
    if model == "Albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2', return_dict=True)
    else:
        raise Exception(f"This model type ({model}) is not known")

    # tokenize the raw input
    tokenized_input = tokenizer(inp, return_tensors="pt")

    # return pytorch tensors from the tokenized input
    output = model(**tokenized_input)

    # retrieve the embedding from the model
    embedding = output.last_hidden_state

    return embedding




data = []
with open("Data/MIMICS-Click.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        data.append(line)

headers = data[0]
data = data[1:]


dataloader = []

for i, (query, question, op1, op2, op3, op4, op5,
        impression_lvl, engagement_lvl, ccp1, ccp2,
        ccp3, cpp4, ccp5) in enumerate(data):

    # convert to right types
    impression_lvl = {"low": 1, "medium": 2, "high": 3}[impression_lvl]
    engagement_lvl = torch.Tensor([int(engagement_lvl)]).int()
    ccp1, ccp2, ccp3, cpp4, ccp5 = float(ccp1), float(ccp2), float(ccp3), float(cpp4), float(ccp5)

    query = get_embedding(query)
    question = get_embedding(question)

    breakpoint()

    nr_options = len([op for op in [op1, op2, op3, op4, op5] if op])

    batch_tensor = (query, question, nr_options, impression_lvl)

    # all datatypes must be converted to torch tensors,
    dataloader.append((batch_tensor, engagement_lvl))

    breakpoint()




# save the dataloader
with open("Data/dataloader.p", "wb") as f:
    pickle.dump(dataloader, f)