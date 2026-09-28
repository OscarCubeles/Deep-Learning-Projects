#!/bin/bash

###

#SBATCH --qos=debug

###

#SBATCH --cpus-per-task=40



#SBATCH --time=2-00:00:00               # acc_bscls wallclock 48h / acc_debug wallclock 2h

###

#SBATCH --job-name="test_mnist-keras-gpu"

#SBATCH --chdir=.

#SBATCH --output=test_mnist-keras-gpu_%j.out

#SBATCH --error=test_mnist-keras-gpu_%j.err

###

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

python train_open_set.py