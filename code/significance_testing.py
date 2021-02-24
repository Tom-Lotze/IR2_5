# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-10-15 15:09
# @Last Modified by:   TomLotze
# @Last Modified time: 2021-02-24 20:36

from scipy.stats import ttest_rel
import numpy as np
import torch
import pickle as pkl
from ranknet import RankNet

# difference between classification and regression on test set

np.random.seed(42)
regression=False
ndcg = True
mrr = True
##### REGRESSION VS. CLASSIFICATION
if regression:
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


##### RANKER ABLATION STUDY
if ndcg:
    with open(f"Predictions/ranker_ndcgs_test_ranking_Adam_0.001_0.001_0_32, 16_0, 0, 0_True_5_False.p", "rb") as f:
        ranker_false = pkl.load(f)

    with open(f"Predictions/ranker_ndcgs_test_ranking_Adam_0.001_0.001_0_32, 16_0, 0, 0_True_5_True.p", "rb") as f:
        ranker_true = pkl.load(f)

    if np.isnan(np.array(ranker_false)).any():
        print("nan found")
    if np.isnan(np.array(ranker_true)).any():
        print("nan found")

    print(len(ranker_false), len(ranker_true))
    # print(ranker_true[-20:])
    # print(ranker_false[-20:])
    assert len(ranker_false) == len(ranker_true)

    t_stat, p_value = ttest_rel(ranker_false, ranker_true)


    print(f"p-value: {p_value}")


if mrr:
    with open(f"Predictions/ranker_MRR_test_ranking_Adam_0.001_0.001_0_32, 16_0, 0, 0_True_5_False.p", "rb") as f:
        ranker_false = pkl.load(f)

    with open(f"Predictions/ranker_MRR_test_ranking_Adam_0.001_0.001_0_32, 16_0, 0, 0_True_5_True.p", "rb") as f:
        ranker_true = pkl.load(f)

    if np.isnan(np.array(ranker_false)).any():
        print("nan found")
    if np.isnan(np.array(ranker_true)).any():
        print("nan found")

    print(len(ranker_false), len(ranker_true))
    # print(ranker_true[-20:])
    # print(ranker_false[-20:])
    assert len(ranker_false) == len(ranker_true)

    t_stat, p_value = ttest_rel(ranker_false, ranker_true)


    print(f"p-value: {p_value}")












