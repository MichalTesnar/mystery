#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=regular

module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate

python3 main_auto.py THRESHOLD 0.0001 &
python3 main_auto.py THRESHOLD 0.0033 &
python3 main_auto.py THRESHOLD 0.0057 &
python3 main_auto.py THRESHOLD 0.0078 &
python3 main_auto.py THRESHOLD 0.0099 &
python3 main_auto.py THRESHOLD 0.0121 &
python3 main_auto.py THRESHOLD 0.0147 &
python3 main_auto.py THRESHOLD 0.0180 &
python3 main_auto.py THRESHOLD 0.0229

wait

module load git
git config --global user.email "michal.tesnar007@gmail.com"
git config --global user.name "MichalTesnar"
git add --a
git commit -m "THRESHOLD different values"
git push

deactivate
