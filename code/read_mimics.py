# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-09-07 16:59
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-09-07 17:03


import csv

data = []
with open("Data/MIMICS-Click.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        data.append(line)

