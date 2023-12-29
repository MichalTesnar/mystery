#!/bin/bash

arguments=("FIFO" "FIRO" "RIRO" "THRESHOLD" "GREEDY" "THRESHOLD_GREEDY" "OFFLINE")
start_port=8000

for ((i=0; i<${#arguments[@]}; i++))
do
    current_port=$((start_port + i))
    arg="${arguments[i]}"
    sbatch --job-name="tune $arg" tuning_fast.sh $current_port "$arg"
done