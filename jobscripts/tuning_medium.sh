#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --partition=regular

module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate

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
bash other_worker.sh 29 $1
wait
# module load git
# git config --global user.email "michal.tesnar007@gmail.com"
# git config --global user.name "MichalTesnar"
# git add --a
# git commit -m "$1, full data medium"
# git push

# deactivate
