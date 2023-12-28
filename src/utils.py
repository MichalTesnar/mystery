import json
import os
import glob

def find_min_score(folder_path):
    min_score = float('inf')
    min_score_file = None
    for file_path in glob.glob(os.path.join(folder_path, '*', 'trial.json')):
        with open(file_path, 'r') as f:
            data = json.load(f)
            score = data.get('score', float('inf'))
            
            # print(score)
            if score != None:
                # print(data)
                if score < min_score:
                    min_score = score
                    min_score_file = file_path
    # print(f"MIN SCORE WAS {min_score}")
    return min_score_file

def get_params(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        hyperparameters = data.get('hyperparameters', dict())
        return hyperparameters['values']

def get_best_params(folder_path):
    # print(folder_path)
    result = find_min_score(folder_path)
    # print(result)
    params = get_params(result)
    return params

def print_best_params(best_hps):
    layers = best_hps['num_layers']
    units = best_hps['units']
    learning_rate = best_hps['learning_rate']
    batch_size = best_hps['batch_size']
    patience = best_hps['patience']
    print(f"\
    ##### EXTRACTED HYPERPARAMS #####\n\
    layers {layers}\n\
    units {units}\n\
    learning_rate {learning_rate}\n\
    batch_size {batch_size}\n\
    patience {patience}\n\
    #################################")
    
