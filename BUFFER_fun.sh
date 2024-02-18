#!/bin/bash
#SBATCH --time=10-0:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=regular

module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate

python3 main_buffer.py $1 10 &
python3 main_buffer.py $1 25 &
python3 main_buffer.py $1 50 &
python3 main_buffer.py $1 100 &
python3 main_buffer.py $1 200 &
python3 main_buffer.py $1 400

wait

module load git
git config --global user.email "michal.tesnar007@gmail.com"
git config --global user.name "MichalTesnar"
git add --a
git commit -m "$1 Buffer Fun"
git push

deactivate
