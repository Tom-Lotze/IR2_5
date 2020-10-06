import csv
import argparse
import torch
from collections import Counter

FLAGS = None

def run():
    """
    Run the simplest of baselines for the engagement_lvls
    """
    engagement_lvls = []
    with open(FLAGS.folder+FLAGS.filename) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)
        for line in tsvreader:
            engagement_lvls.append(float(line[8]))

    for key, value in Counter(engagement_lvls).items():
        print(f"{key}: {value}")
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
    
    FLAGS, unparsed = parser.parse_known_args()

    run()