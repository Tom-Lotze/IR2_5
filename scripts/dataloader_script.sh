#!/bin/bash
#SBATCH --job-name="dataloader"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_shared_course


module load pre2019
module load Miniconda3
source activate ir2

cp -r $HOME/IR2_5/ $TMPDIR/

cd $TMPDIR/IR2_5

echo "Dataloader started running" | mail $USER

python code/test_cuda_print.py
python code/dataloader.py --old True

cp -r  $TMPDIR/IR2_5/Data/*.p $HOME/IR2_5/dataloader/


echo "dataloader finished" | mail $USER


