#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=regular

module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate

# type of runtime / port
python3 main_auto.py THRESHOLD_GREEDY 0.0016 &
python3 main_auto.py THRESHOLD_GREEDY 0.0023 &
python3 main_auto.py THRESHOLD_GREEDY 0.0036 &
python3 main_auto.py THRESHOLD_GREEDY 0.0056 &
python3 main_auto.py THRESHOLD_GREEDY 0.0075 &
python3 main_auto.py THRESHOLD_GREEDY 0.0096 &
python3 main_auto.py THRESHOLD_GREEDY 0.0120 &
python3 main_auto.py THRESHOLD_GREEDY 0.0156 &
python3 main_auto.py THRESHOLD_GREEDY 0.0228

wait

module load git
git config --global user.email "michal.tesnar007@gmail.com"
git config --global user.name "MichalTesnar"
git add --a
git commit -m "$1 different values"
git push

deactivate
