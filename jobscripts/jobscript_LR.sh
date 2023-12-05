#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=regular

module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate

python3 tunes_LR.py $1 # run the script with different arguments

module load git
git config --global user.email "michal.tesnar007@gmail.com"
git config --global user.name "MichalTesnar"
git add --a
git commit -m "LR results"
git push

deactivate
