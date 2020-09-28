#!/bin/bash
#SBATCH --job-name="Tuning"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:0

module load pre2019
module load Miniconda3
source activate ir2

cp -r $HOME/IR2_5 $TMPDIR/
cd $TMPDIR/IR2_5

mkdir -p Models
mkdir -p Images

echo "starting hyperparam tuning" | mail $USER

declare -a dnn=("256, 128, 32" "2000, 100, 16" "300, 32")
declare -a lrs=("0.001" "0.01" "0.0001")
declare -a wd=("0.00001" "0.01")
declare -a dropouts = ("0.2, 0.1, 0.05" "0.1 0")

for drop in "${dropouts[@]}"
do
  echo "$drop" | mail $USER
  for weight_decay in "${wd[@]}"
  do
    for learning_rate in "${lrs[@]}"
    do
      for units in "${dnn[@]}"
      do
        python code/train_regression.py --nr_epochs 150 --optimizer "Adam"\
         --learning_rate "$learning_rate" --amsgrad "1"\
         --dnn_hidden_units "$units" --weightdecay "$weight_decay"\
         --dropout_percentages "$drop" >> results_tuning.txt
       done
    done
  done
done

cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/
cp results_tuning.txt $HOME/IR2_5/

echo "Dropout search finished, results copied back to home" | mail $USER