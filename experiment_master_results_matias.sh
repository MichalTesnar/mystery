source $HOME/venvs/mystery/bin/activate

nohup python3 main_auto.py "FIFO" &
nohup python3 main_auto.py "FIRO" &
nohup python3 main_auto.py "RIRO" &
nohup python3 main_auto.py "THRESHOLD" &
nohup python3 main_auto.py "GREEDY" &
nohup python3 main_auto.py "THRESHOLD_GREEDY" &

wait