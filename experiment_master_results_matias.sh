source $HOME/venvs/mystery/bin/activate

nohup python3 main_auto.py "FIFO" &
nohup python3 main_auto.py "FIRO" &
nohup python3 main_auto.py "RIRO" &
nohup python3 main_auto.py "THRESHOLD" &
nohup python3 main_auto.py "GREEDY" &
nohup python3 main_auto.py "THRESHOLD_GREEDY" &

# nohup python3 main_auto.py "FIFO" > log_FIFO.out &
# nohup python3 main_auto.py "FIRO" > log_FIRO.out &
# nohup python3 main_auto.py "RIRO" > log_RIRO.out &
# nohup python3 main_auto.py "THRESHOLD" > log_THRESHOLD.out &
# nohup python3 main_auto.py "GREEDY" > log_GREEDY.out &
# nohup python3 main_auto.py "THRESHOLD_GREEDY" > log_THRESHOLD_GREEDY.out &

wait