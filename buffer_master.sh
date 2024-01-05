arguments=("FIFO" "FIRO" "RIRO" "THRESHOLD" "GREEDY" "THRESHOLD_GREEDY" "OFFLINE")

for arg in "${arguments[@]}"
do
    sbatch --job-name="bf $arg" BUFFER_fun.sh "$arg"
done

