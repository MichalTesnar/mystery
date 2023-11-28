# arguments=("FIFO" "FIRO" "RIRO" "THRESHOLD" "GREEDY" "THRESHOLD_GREEDY")
arguments=("THRESHOLD" "THRESHOLD_GREEDY")

module purge

for arg in "${arguments[@]}"
do
    sbatch --job-name="hp sinus $arg" jobscript.sh "$arg" 
done

