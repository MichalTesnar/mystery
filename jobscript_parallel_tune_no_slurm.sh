#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=regular

# module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/mystery/bin/activate


# python3 online_learning_tuning.py $1 # run the script with different arguments
bash main_worker.sh $1 &
bash other_worker.sh 0 $1 &
bash other_worker.sh 1 $1 &
bash other_worker.sh 2 $1 &
bash other_worker.sh 3 $1 &
bash other_worker.sh 4 $1 &
bash other_worker.sh 5 $1 &
bash other_worker.sh 6 $1

# module load git
# git config --global user.email "michal.tesnar007@gmail.com"
# git config --global user.name "MichalTesnar"
# git add --a
# git commit -m "FULL DATA"
# git push

deactivate
