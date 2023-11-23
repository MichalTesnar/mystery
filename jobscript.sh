#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --partition=regular

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate

python3 online_learning_tuning.py

deactivate
