#!/bin/bash
#SBATCH --job-name="sanity check"
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

echo "Regression started running" | mail $USER

python code/sanity_check.py --nr_epochs 100 --weightdecay 0.0001 --optimizer Adam --amsgrad 1

cp -r Models/* $HOME/IR2_5/Models/
cp -r Images/* $HOME/IR2_5/Images/

echo "Regression finished" | mail $USER


