#!/bin/bash
#SBATCH --job-name="dataloader"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_shared_course

source activate ir2


cp -r $HOME/IR2_5/ $TMPDIR/

cd $TMPDIR/IR2_5

echo "Dataloader started running" | mail $USER
python code/dataloader_sentence.py

cp -r  $TMPDIR/IR2_5/Data/ $HOME/IR2_5/dataloader/


echo "dataloader finished" | mail $USER


