from torch.utils.data import Dataset
import itertools
import torch
import numpy as np

class RankDataSet(Dataset):
    def __init__(self, dataset, use_preds):
        """
        Save the dataset class in the RankDataSet for use with dataloader
        """
        self.dataset = dataset
        self.use_preds = use_preds

        self.labels = torch.tensor(self.dataset.engagement_lvls).reshape(-1,1)
        
        self.n_answers = [sum(1 for i in answer if i != "") for answer in self.dataset.answers]
        self.n_answers = torch.tensor(n_answers).reshape(-1,1)
        self.n_answers = self.n_answers - self.n_answers.mean()

        self.avg_answer_len = [np.mean([len(i) for i in answer]) for answer in self.dataset.answers]
        self.avg_answer_len = torch.tensor(avg_answer_len).reshape(-1,1)
        self.avg_answer_len = self.avg_answer_len - self.avg_answer_len.mean()

        self.query_len = [len(query) for query in self.dataset.query]
        self.query_len = torch.tensor(query_len).reshape(-1,1)
        self.query_len = self.query_len - self.query_len.mean()

        self.question_len = [len(question) for question in self.dataset.questions]
        self.question_len = torch.tensor(question_len).reshape(-1,1)
        self.question_len = self.question_len - self.question_len.mean()

    def __len__(self):
        """
        Overwrite of the len function to get the number of items in the dataset
        """
        return len(self.dataset.ranges)

    def __getitem__(self, index):
        """
        Overwrite of the get_item function to return a batch through a Dataloader
        """
        # TODO concat all the datapoints
        indices = self.dataset.ranges[index]

        n_answers = self.n_answers[indices]
        query_len = self.query_len[indices]
        question_len = self.question_len[indices]
        avg_answer_len = self.avg_answer_len[indices]
        labels = self.labels[indices]

        # click_probs = [self.dataset.click_probs[i] for i in indices]
        # click_probs = torch.tensor(click_probs).reshape(-1,5)

        vectors = torch.cat((n_answers, query_len, question_len, avg_answer_len, labels), dim=1)

        if self.use_preds:
            preds = [self.dataset.predictions[i] for i in indices]
            preds = torch.tensor(preds).reshape(-1,1)
            vectors = torch.cat((vectors, preds), dim=1)

        return vectors, labels