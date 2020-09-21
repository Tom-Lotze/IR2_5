#!/bin/bash
#SBATCH --job-name="regression"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_shared_course

source activate ir2

cp -r $HOME/IR2_5/dataloader $TMPDIR/

cd $TMPDIR/IR2_5
mkdir $TMPDIR/Models

echo "Regression started running" | mail $USER

python code/train_regression.py -nr_epochs 10

cp -r $TMPDIR/Models/ $HOME/IR2_5/

echo "Regression finished" | mail $USER


