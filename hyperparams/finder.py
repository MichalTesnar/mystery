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
    return min_score_file

def get_params(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        hyperparameters = data.get('hyperparameters', dict())
        return hyperparameters['values']

def get_best_params(folder_path):
    print(folder_path)
    result = find_min_score(folder_path)
    print(result)
    params = get_params(result)
    return params
    
