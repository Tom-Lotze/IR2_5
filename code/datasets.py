from torch.utils.data import Dataset
import itertools

class RankDataSet(Dataset):
    def __init__(self, dataset):
        """
        Save the dataset class in the RankDataSet for use with dataloader
        """
        self.dataset = dataset

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
        labels = self.dataset.engagement_lvls[indices]


        return vectors, labels