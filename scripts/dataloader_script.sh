#!/bin/bash
#SBATCH --job-name="dataloader"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_shared_course

source activate ir2

echo "Dataloader started running" | mail $USER
python code/dataloader_sentence.py

echo "dataloader finished" | mail $USER


