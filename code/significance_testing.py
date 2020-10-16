# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-10-15 15:09
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-10-15 22:42

from scipy.stats import ttest_rel
import numpy as np
import torch
import pickle as pkl

# difference between classification and regression on test set

for reduced_classes in [True, False]:
    for impression in [True, False]:
        for embedder in ["Bert", "TFIDF"]:
            print(f"Embedder: {embedder}\nreduced_class: {reduced_classes}\nimpression:{impression}")

            with open(f"Data/regression_test_preds{embedder}_{reduced_classes}_{impression}.pt", "rb") as f:
                test_pred_regression = pkl.load(f)

            with open(f"Data/classification_test_preds{embedder}_{reduced_classes}_{impression}.pt", "rb") as f:
                test_pred_classification = pkl.load(f)

            breakpoint()
            # todo: check and assert shapes
            # todo: round the classification preds to nearest integer

            t_stat, p_value = ttest_rel(test_pred_regression, test_pred_classification)

            print(f"p-value: {p_value}")