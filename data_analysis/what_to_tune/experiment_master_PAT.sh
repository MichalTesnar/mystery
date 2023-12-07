arguments=("FIFO" "FIRO" "RIRO" "THRESHOLD" "GREEDY" "THRESHOLD_GREEDY")

module purge

for arg in "${arguments[@]}"
do
    sbatch --job-name="PAT $arg" jobscript_PAT.sh "$arg" 
done

