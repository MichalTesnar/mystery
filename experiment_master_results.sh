arguments=("FIFO" "FIRO" "RIRO" "THRESHOLD" "GREEDY" "THRESHOLD_GREEDY" "OFFLINE")

for arg in "${arguments[@]}"
do
    sbatch --job-name="run $arg" jobscript_results.sh "$arg"
done

