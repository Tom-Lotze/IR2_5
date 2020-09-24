#!/bin/bash
#SBATCH --job-name="regression"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_shared_course

module load pre2019
module load Miniconda3
source activate ir2

cp -r $HOME/IR2_5 $TMPDIR/

cd $TMPDIR/IR2_5
mkdir -p Models
mkdir -p Images

echo "Regression started running" | mail $USER

python code/train_regression.py --nr_epochs 200 --weightdecay 0.02 --optimizer AdamW --amsgrad 1

cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/

echo "Regression finished" | mail $USER


