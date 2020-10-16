from torch.utils.data import Dataset
import itertools
import torch

class RankDataSet(Dataset):
    def __init__(self, dataset, use_preds):
        """
        Save the dataset class in the RankDataSet for use with dataloader
        """
        self.dataset = dataset
        self.use_preds = use_preds

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
        labels = [self.dataset.engagement_lvls[i] for i in indices]
        labels = torch.tensor(labels).reshape(-1,1)
        
        answers = [self.dataset.answers[i] for i in indices]
        n_answers = [sum(1 for i in answer if i != "") for answer in answers]
        n_answers = torch.tensor(n_answers).reshape(-1,1)

        query_len = [len(self.dataset.queries[i]) for i in indices]
        query_len = torch.tensor(query_len).reshape(-1,1)

        question_len = [len(self.dataset.questions[i]) for i in indices]
        question_len = torch.tensor(question_len).reshape(-1,1)

        click_probs = [self.dataset.click_probs[i] for i in indices]
        click_probs = torch.tensor(click_probs).reshape(-1,5)

        vectors = torch.cat((n_answers, query_len, question_len, click_probs), dim=1)

        if self.use_preds:
            preds = [self.dataset.predictions[i] for i in indices]
            preds = torch.tensor(preds).reshape(-1,1)
            vectors = torch.cat((vectors, preds), dim=1)

        return vectors, labels