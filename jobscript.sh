#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=regular

module purge
module load Python/3.9.6-GCCcore-11.2.0
module load git

git config --global user.email "michal.tesnar007@gmail.com"
git config --global user.name "MichalTesnar"

source $HOME/venvs/mystery/bin/activate

python3 online_learning_tuning.py $1 # run the script with different arguments

git add --a
git commit -m "Pushing online_learning_tuning.py results"
git push


deactivate
