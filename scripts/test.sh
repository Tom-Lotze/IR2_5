#!/bin/bash
#SBATCH --job-name="cuda_test"
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_shared_course

module load pre2019
module load Miniconda3
source activate ir2
conda list torch

cp -r $HOME/IR2_5 $TMPDIR/

cd $TMPDIR/IR2_5

python test_cuda_lgpu0008.py