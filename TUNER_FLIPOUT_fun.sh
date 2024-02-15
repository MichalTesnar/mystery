#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=regular

module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate

python3 main_auto.py FIFO 0.0 &
python3 main_auto.py FIFO 0.1 &
python3 main_auto.py FIFO 0.2 &
python3 main_auto.py FIFO 0.3 &
python3 main_auto.py FIFO 0.4 &
python3 main_auto.py FIFO 0.5 &
python3 main_auto.py FIFO 0.6 &
python3 main_auto.py FIFO 0.7 &
python3 main_auto.py FIFO 0.8 &
python3 main_auto.py FIFO 0.9 &
python3 main_auto.py FIFO 1.0

wait

module load git
git config --global user.email "michal.tesnar007@gmail.com"
git config --global user.name "MichalTesnar"
git add --a
git commit -m "THRESHOLD_GREEDY different values"
git push

deactivate
