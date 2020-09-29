#!/bin/bash
#SBATCH --job-name="regression"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:0

module load pre2019
module load Miniconda3
source activate ir2
conda list torch

cp -r $HOME/IR2_5 $TMPDIR/

cd $TMPDIR/IR2_5
conda list torch
pwd
mkdir -p Models
mkdir -p Images

echo "Regression started running" | mail $USER

python code/train_regression.py --nr_epochs 0 --weightdecay 0.02 --optimizer Adam --amsgrad 1

cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/

echo "Regression finished" | mail $USER


