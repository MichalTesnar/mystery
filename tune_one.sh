#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=regular

module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate

python3 online_learning_tuning.py $1

deactivate

