arguments=("FIFO" "FIRO" "FIRO" "THRESHOLD" "GREEDY" "THRESHOLD_GREEDY")

for arg in "${arguments[@]}"
do
    sbatch jobscript.sh arg
done