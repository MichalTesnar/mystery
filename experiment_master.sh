arguments=("FIFO" "FIRO" "FIRO" "THRESHOLD" "GREEDY" "THRESHOLD_GREEDY" "RIRO")

module purge

for arg in "${arguments[@]}"
do
    sbatch --job-name="hp sinus $arg" jobscript.sh "$arg" 
done

