# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-10-15 15:09
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-10-16 14:01

from scipy.stats import ttest_rel
import numpy as np
import torch
import pickle as pkl

# difference between classification and regression on test set

np.random.seed(42)



for reduced_classes in [True, False]:
    for impression in [True, False]:
        for embedder in ["Bert", "TFIDF"]:
            print(f"\n\nEmbedder: {embedder}\nreduced_class: {reduced_classes}\nimpression:{impression}")

            with open(f"Predictions/regression_test_preds{embedder}_{reduced_classes}_{impression}.pt", "rb") as f:
                test_pred_regression = pkl.load(f)

            with open(f"Predictions/classification_test_preds{embedder}_{reduced_classes}_{impression}.pt", "rb") as f:
                test_pred_classification = pkl.load(f)

            test_pred_regression = [round(i) for i in test_pred_regression]

            assert len(test_pred_classification) == len(test_pred_regression)

            if reduced_classes:
                random_model = np.random.randint(0, 2, len(test_pred_classification))
                t, p = ttest_rel(test_pred_classification, random_model)
                print(f"p-value random model: {p}")

            t_stat, p_value = ttest_rel(test_pred_regression, test_pred_classification)

            print(f"p-value: {p_value}")