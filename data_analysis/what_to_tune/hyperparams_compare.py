import os
import json
import matplotlib.pyplot as plt
import numpy as np

def read_score_from_json_directory(directory_path):
    scores = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isdir(file_path):
            for inner_file in os.listdir(file_path):
                inner_file_path = os.path.join(file_path, inner_file)
                if inner_file.endswith(".json"):
                    with open(inner_file_path, 'r') as file:
                        data = json.load(file)
                    score = data.get('score')
                    vals = data.get("hyperparameters")["values"]
                    par_value = list(vals.values())[0]
                    if par_value == 0.1:
                        continue
                    if score is not None:
                        if par_value in scores:
                            scores[par_value].append(score)
                        else:
                            scores[par_value] = [score]
    return scores

def plot_results(results, param):
    
    x = 0  # the label locations
    width = 1  # the width of the bars
    multiplier = 0
    multiplier2 = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(20, 12))
    labels = []
    indices = []
    for method, scores in results.items():
        for value in scores:
            given_results = np.array(results[method][value])
            mean = np.mean(given_results)
            std = np.std(given_results)
            offset = width * (multiplier + multiplier2)
            ax.bar(x + offset, mean, width, label=f"{method} {value}", yerr=std, capsize=5)
            indices.append(x + offset)
            labels.append(f"{method[:3]} {value}")
            multiplier += 1
        labels.append("")
        offset = width * (multiplier + multiplier2)
        indices.append(x + offset)
        multiplier2 += 1
    name = "LOL"
    if param == "BS":
        name = "Batch Sizes"
    elif param == "LR":
        name = "Learning Rate"
    elif param == "PAT":
        name = "Patience"
    ax.set_title(f'{name}')
    ax.legend(loc='upper left', ncols=3)
    plt.xticks(indices, labels)
    plt.savefig(f"hyperparams_{name}")
    plt.show()

if __name__ == "__main__":
    params = ["BS", "LR", "PAT"]
    methods = ["FIRO", "RIRO", "FIFO", "THRESHOLD", "GREEDY", "THRESHOLD_GREEDY"]
    script_dir = os.path.dirname(__file__)
    
    for param in params:
        results = {}
        for method in methods:
            directory_path = f"hyperparams/Tuning {param} {method}"
            full_directory_path = os.path.join(script_dir, directory_path)
            
            caught = read_score_from_json_directory(full_directory_path)
            results[method] = dict(sorted(caught.items()))
        plot_results(results, param)

        