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
mkdir -p Predictions

echo "Regression started running" | mail $USER

for emb in "Bert"
do
  for imp in 0 1
  do
    for rc in 0 1
    do
      python code/train_regression.py --embedder "$emb" --impression "$imp" --reduced_classes "$rc"
    done
  done
done
cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/
cp -r Predictions/* $HOME/IR2_5/Predictions/

echo "Regression finished" | mail $USER


