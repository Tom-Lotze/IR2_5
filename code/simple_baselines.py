import csv
import argparse
import torch
from collections import Counter
import numpy as np

FLAGS = None

def run():
    """
    Run the simplest of baselines for the engagement_lvls
    """
    np.random.seed(40)

    engagement_lvls = []
    
    with open(FLAGS.folder+FLAGS.filename) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)

        for line in tsvreader:
            # Skip line if impression level was low
            if FLAGS.impression and line[7] == "low":
                continue
            # Add values to the data lists
            engagement_lvls.append(float(line[8]))

    # Attempt to fix class imbalance assuming 0 is to large
    if FLAGS.balance:
        # Index the locations of zeros and non-zeros
        engagement_lvls = np.array(engagement_lvls)
        zero_indices = np.where(engagement_lvls == 0)[0]
        non_zero_indices = np.where(engagement_lvls != 0)[0]

        # Get the median size of the engagement levels
        median_size = int(np.median(list(Counter(engagement_lvls).values())))
        
        # Return the to be used indices
        sampled_indices = np.random.choice(zero_indices, median_size, replace=False)
        indices = np.concatenate((sampled_indices, non_zero_indices))

        # Update datalist based on indices
        engagement_lvls = [engagement_lvls[i] for i in indices]

    print("Engagement levels:", Counter(engagement_lvls))

    engagement_lvls = torch.tensor(engagement_lvls)
    mean = torch.full_like(engagement_lvls, torch.mean(engagement_lvls))
    median = torch.full_like(engagement_lvls, torch.median(engagement_lvls))
    mode = torch.full_like(engagement_lvls, torch.mode(engagement_lvls)[0])

    MSE_loss = torch.nn.MSELoss()

    MSE_mean = MSE_loss(mean, engagement_lvls)
    MSE_median = MSE_loss(median, engagement_lvls)
    MSE_mode = MSE_loss(mode, engagement_lvls)

    print(f"MSE mean Loss: {MSE_mean}")
    print(f"MSE median Loss: {MSE_median}")
    print(f"MSE mode Loss: {MSE_mode}")

    CE_loss = torch.nn.CrossEntropyLoss()

    engagement_lvls = engagement_lvls.long()
    mode_one_hot = torch.nn.functional.one_hot(mode.long(), 11).float()
    CE_mode = CE_loss(mode_one_hot, engagement_lvls)

    print(f"CE mode Loss: {CE_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='Data/', 
                        help='Folder where the data is located')
    parser.add_argument('--filename', type=str, default="MIMICS-Click.tsv",
                        help='Filename of the data')
    parser.add_argument('--balance', type=bool, default=True, 
                        help='Balance the data by fixing the distributions')
    parser.add_argument('--impression', type=bool, default=True,
                        help='Use only the most shown clarification panes')
    parser.add_argument('--bins', type=int, default=11,
                        help='Number of classes to consider')
    FLAGS, unparsed = parser.parse_known_args()

    run()