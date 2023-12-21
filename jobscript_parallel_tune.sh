#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=59
#SBATCH --partition=regular

module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate


# python3 online_learning_tuning.py $1 # run the script with different arguments
bash main_worker.sh $1 &
bash other_worker.sh 0 $1 &
bash other_worker.sh 1 $1 &
bash other_worker.sh 2 $1 &
bash other_worker.sh 3 $1 &
bash other_worker.sh 4 $1 &
bash other_worker.sh 5 $1 &
bash other_worker.sh 6 $1 &
bash other_worker.sh 7 $1 &
bash other_worker.sh 8 $1 &
bash other_worker.sh 9 $1 &
bash other_worker.sh 10 $1 &
bash other_worker.sh 11 $1 &
bash other_worker.sh 12 $1 &
bash other_worker.sh 13 $1 &
bash other_worker.sh 14 $1 &
bash other_worker.sh 15 $1 &
bash other_worker.sh 16 $1 &
bash other_worker.sh 17 $1 &
bash other_worker.sh 18 $1 &
bash other_worker.sh 19 $1 &
bash other_worker.sh 20 $1 &
bash other_worker.sh 21 $1 &
bash other_worker.sh 22 $1 &
bash other_worker.sh 23 $1 &
bash other_worker.sh 24 $1 &
bash other_worker.sh 25 $1 &
bash other_worker.sh 26 $1 &
bash other_worker.sh 27 $1 &
bash other_worker.sh 28 $1 &
bash other_worker.sh 29 $1 &
bash other_worker.sh 30 $1 &
bash other_worker.sh 31 $1 &
bash other_worker.sh 32 $1 &
bash other_worker.sh 33 $1 &
bash other_worker.sh 34 $1 &
bash other_worker.sh 35 $1 &
bash other_worker.sh 36 $1 &
bash other_worker.sh 37 $1 &
bash other_worker.sh 38 $1 &
bash other_worker.sh 39 $1 &
bash other_worker.sh 40 $1 &
bash other_worker.sh 41 $1 &
bash other_worker.sh 42 $1 &
bash other_worker.sh 43 $1 &
bash other_worker.sh 44 $1 &
bash other_worker.sh 45 $1 &
bash other_worker.sh 46 $1 &
bash other_worker.sh 47 $1 &
bash other_worker.sh 48 $1 &
bash other_worker.sh 49 $1 &
bash other_worker.sh 50 $1 &
bash other_worker.sh 51 $1 &
bash other_worker.sh 52 $1 &
bash other_worker.sh 53 $1 &
bash other_worker.sh 54 $1 &
bash other_worker.sh 55 $1 &
bash other_worker.sh 56 $1 &
bash other_worker.sh 57 $1 &
bash other_worker.sh 58 $1

module load git
git config --global user.email "michal.tesnar007@gmail.com"
git config --global user.name "MichalTesnar"
git add --a
git commit -m "FULL DATA, $1"
git push

deactivate
