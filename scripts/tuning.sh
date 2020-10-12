#!/bin/bash
#SBATCH --job-name="Tuning"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --partition=gpu_shared_course

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

declare -a dnn=("2000, 100, 16" "300, 32" "248")
declare -a dropouts=("0.3, 0.2, 0.1" "0.3, 0.1" "0.1")
declare -a lrs=("0.001" "0.0001")
declare -a embedders=("Bert" "TFIDF")


declare len=${#dropouts[@]}

for optimizer in 'Adam' 'SGD'
do
  for emb in "${embedders[@]}"
  do
    for learning_rate in "${lrs[@]}"
    do
      for i in $(seq 0 "$len")
      do
        python code/train_regression.py --nr_epochs 80 --optimizer $optimizer\
         --learning_rate "$learning_rate" --verbose 0 --embedder emb\
         --dnn_hidden_units "${dnn[$i]}" --dropout_percentages "${dropouts[$i]}"
       done
     done
  done
done

cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/

echo "Param search finished, results copied back to home" | mail $USER
