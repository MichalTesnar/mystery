
source $HOME/venvs/mystery/bin/activate

# type of runtime / port
nohup bash main_worker.sh FIFO 8005 &
nohup bash other_worker.sh 0 FIFO 8005 &
nohup bash other_worker.sh 1 FIFO 8005 &
nohup bash other_worker.sh 2 FIFO 8005 &
nohup bash other_worker.sh 3 FIFO 8005 &
nohup bash other_worker.sh 4 FIFO 8005 &
nohup bash other_worker.sh 5 FIFO 8005 &
nohup bash other_worker.sh 6 FIFO 8005 &
nohup bash other_worker.sh 7 FIFO 8005 &
nohup bash other_worker.sh 8 FIFO 8005 &
nohup bash other_worker.sh 9 FIFO 8005 &
nohup bash other_worker.sh 10 FIFO 8005 &
nohup bash other_worker.sh 11 FIFO 8005 &
nohup bash other_worker.sh 12 FIFO 8005 &
nohup bash other_worker.sh 13 FIFO 8005 &
nohup bash other_worker.sh 14 FIFO 8005 &
nohup bash other_worker.sh 15 FIFO 8005 &
nohup bash other_worker.sh 16 FIFO 8005 &
nohup bash other_worker.sh 17 FIFO 8005 &
nohup bash other_worker.sh 18 FIFO 8005 &
nohup bash other_worker.sh 19 FIFO 8005 &
nohup bash other_worker.sh 20 FIFO 8005 &
wait
