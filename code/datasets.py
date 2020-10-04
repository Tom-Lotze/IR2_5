from torch.utils.data import Dataset
import itertools

class RankDataSet(Dataset):
    def __init__(self, dataset):

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.ranges)

    def __getitem__(self, index):
        # TODO
        indices = self.dataset.ranges[index]
        labels = self.dataset.engagement_lvls[indices]
        

        return vectors, labels