export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="$2"
python3 online_learning_tuning.py $1