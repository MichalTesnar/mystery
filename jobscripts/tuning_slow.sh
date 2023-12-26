#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=regular
#SBATCH --job-name=$1

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
bash other_worker.sh 10 $1
wait
# module load git
# git config --global user.email "michal.tesnar007@gmail.com"
# git config --global user.name "MichalTesnar"
# git add --a
# git commit -m "$1, full data 10"
# git push

# deactivate
