arguments=("FIFO" "FIRO" "FIRO" "THRESHOLD" "GREEDY" "THRESHOLD_GREEDY")

module purge

for arg in "${arguments[@]}"
do
    sbatch --job-name="hp sinus $arg" jobscript.sh "$arg" 
done

module load git
git config --global user.email "michal.tesnar007@gmail.com"
git config --global user.name "MichalTesnar"
git add --a
git commit -m "Pushing online_learning_tuning.py results"
git push