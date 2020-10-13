#!/bin/bash
#SBATCH --job-name="regression"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_shared_course

module load pre2019
module load Miniconda3
source activate ir2

cp -r $HOME/IR2_5 $TMPDIR/

cd $TMPDIR/IR2_5
mkdir -p Models
mkdir -p Images

echo "Regression started running" | mail $USER

python code/test_cuda_print.py
python code/train_regression.py --nr_epochs 100 --weightdecay 0.0001 \
    --optimizer SGD --momentum 0.9 --amsgrad 1 --batchnorm 1 \
    --dropout_probs "0.3, 0.05" --embedder 'TFIDF'

cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/

echo "Regression finished" | mail $USER


