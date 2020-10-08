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

echo "Classification started running" | mail $USER

python code/train_classifier.py --nr_epochs 100 --weightdecay 0.0001 --optimizer Adam --amsgrad 1 \
	 --batchnorm 1 --dropout_probs "0.3, 0.05"

cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/

echo "Classification training finished" | mail $USER


