#!/bin/bash
#SBATCH -p gpu

module load 2020
module load Python

source spc/bin/activate spc

python $HOME/spc/tf_pc2_batch.py
