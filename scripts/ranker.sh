#!/bin/bash
#SBATCH --job-name="ranker"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_shared_course

module load pre2019
module load Miniconda3
source activate ir2

cp -r $HOME/IR2_5 $TMPDIR/

cd $TMPDIR/IR2_5
mkdir -p Models
mkdir -p Images
mkdir -p Predictions

echo "Ranker started running" | mail $USER

python code/test_cuda_print.py
python code/train_ranknet.py --plotting 1 --use_preds 0 --nr_epochs 5 --use_preds 1

cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/
cp -r Predictions/* $HOME/IR2_5/Predictions/

echo "Ranker finished" | mail $USER


