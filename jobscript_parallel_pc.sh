#!/bin/bash

bash main_worker.sh $1 &
bash other_worker.sh 0 $1 &
bash other_worker.sh 1 $1 &
bash other_worker.sh 2 $1 &
bash other_worker.sh 3 $1 &
bash other_worker.sh 4 $1

