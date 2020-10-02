#!/bin/bash
#SBATCH --job-name="Tuning"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge
module load pre2019
module load Miniconda3
source activate ir2

cp -r $HOME/IR2_5 $TMPDIR/
cd $TMPDIR/IR2_5

mkdir -p Models
mkdir -p Images


echo "Results file" > results_tuning.txt

echo "starting hyperparam tuning" | mail $USER

declare -a dnn=("256, 128, 32" "2000, 100, 16" "300, 32")
declare -a lrs=("0.001" "0.01" "0.0001")
declare -a wd=("0.00001" "0.01")

for optimizer in 'Adam' 'AdamW' 'SGD' 'RMSprop'
do
  echo $optimizer | mail $USER
  for weight_decay in "${wd[@]}"
  do
    for learning_rate in "${lrs[@]}"
    do
      for units in "${dnn[@]}"
      do
        python code/train_regression.py --nr_epochs 80 --optimizer $optimizer\
         --learning_rate "$learning_rate" --amsgrad "1"\
         --dnn_hidden_units "$units" --weightdecay "$weight_decay"\
         >> results_tuning.txt
       done
    done
  done
done

cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/
cp results_tuning.txt $HOME/IR2_5/

echo "Param search finished, results copied back to home" | mail $USER
