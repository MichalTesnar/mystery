#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=regular

module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate

# type of runtime / port
bash main_worker.sh $1 $2 &
bash other_worker.sh 0 $1 $2 &
bash other_worker.sh 1 $1 $2 &
bash other_worker.sh 2 $1 $2 &
bash other_worker.sh 3 $1 $2 &
bash other_worker.sh 4 $1 $2 &
bash other_worker.sh 5 $1 $2 &
bash other_worker.sh 6 $1 $2 &
bash other_worker.sh 7 $1 $2 &
bash other_worker.sh 8 $1 $2 &
bash other_worker.sh 9 $1 $2 &
bash other_worker.sh 10 $1 $2 &
bash other_worker.sh 11 $1 $2 &
bash other_worker.sh 12 $1 $2 &
bash other_worker.sh 13 $1 $2 &
bash other_worker.sh 14 $1 $2 &
bash other_worker.sh 15 $1 $2 &
bash other_worker.sh 16 $1 $2 &
bash other_worker.sh 17 $1 $2 &
bash other_worker.sh 18 $1 $2 &
bash other_worker.sh 19 $1 $2 &
bash other_worker.sh 20 $1 $2 &
bash other_worker.sh 21 $1 $2 &
bash other_worker.sh 22 $1 $2 &
bash other_worker.sh 23 $1 $2 &
bash other_worker.sh 24 $1 $2 &
bash other_worker.sh 25 $1 $2 &
bash other_worker.sh 26 $1 $2 &
bash other_worker.sh 27 $1 $2 &
bash other_worker.sh 28 $1 $2 &
bash other_worker.sh 29 $1 $2 &
bash other_worker.sh 30 $1 $2 &
bash other_worker.sh 31 $1 $2 &
bash other_worker.sh 32 $1 $2 &
bash other_worker.sh 33 $1 $2 &
bash other_worker.sh 34 $1 $2 &
bash other_worker.sh 35 $1 $2 &
bash other_worker.sh 36 $1 $2 &
bash other_worker.sh 37 $1 $2 &
bash other_worker.sh 38 $1 $2 &
bash other_worker.sh 39 $1 $2 &
bash other_worker.sh 40 $1 $2 &
bash other_worker.sh 41 $1 $2 &
bash other_worker.sh 42 $1 $2 &
bash other_worker.sh 43 $1 $2 &
bash other_worker.sh 44 $1 $2 &
bash other_worker.sh 45 $1 $2 &
bash other_worker.sh 46 $1 $2 &
bash other_worker.sh 47 $1 $2 &
bash other_worker.sh 48 $1 $2 &
bash other_worker.sh 49 $1 $2 &
bash other_worker.sh 50 $1 $2 &
bash other_worker.sh 51 $1 $2 &
bash other_worker.sh 52 $1 $2 &
bash other_worker.sh 53 $1 $2 &
bash other_worker.sh 54 $1 $2 &
bash other_worker.sh 55 $1 $2 &
bash other_worker.sh 56 $1 $2 &
bash other_worker.sh 57 $1 $2 &
bash other_worker.sh 58 $1 $2 &
bash other_worker.sh 59 $1 $2 

module load git
git config --global user.email "michal.tesnar007@gmail.com"
git config --global user.name "MichalTesnar"
git add --a
git commit -m "$1, full data fast"
git push

deactivate
