arguments=("FIFO" "FIRO" "RIRO" "THRESHOLD" "GREEDY" "THRESHOLD_GREEDY")

module purge

for arg in "${arguments[@]}"
do
    sbatch --job-name="BS $arg" jobscript_BS.sh "$arg" 
done

