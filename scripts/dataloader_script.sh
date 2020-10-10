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

python code/dataloader.py --expanded False --balance True --impression True \
    --embedder TFIDF

cp -r  $TMPDIR/IR2_5/Data/*.p $HOME/IR2_5/Data/


echo "dataloader finished, results copied to Data folder" | mail $USER


