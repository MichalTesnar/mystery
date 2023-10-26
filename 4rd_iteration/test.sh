#!/bin/bash

# Loop 10 times
for ((i=1; i<=10; i++)); do
    echo "Running iteration $i"
    python3 main.py 
done
