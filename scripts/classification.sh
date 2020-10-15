#!/bin/bash
#SBATCH --job-name="regression"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module load pre2019
module load Miniconda3
source activate ir2

cp -r $HOME/IR2_5 $TMPDIR/

cd $TMPDIR/IR2_5
mkdir -p Models
mkdir -p Images
mkdir -p Predictions

echo "Classification started running" | mail $USER

python code/train_classifier.py

cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/
cp -r Predictions/* $HOME/IR2_5/Predictions/

echo "Classification training finished" | mail $USER


