import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import sys

sys.path.append('..')

# EXPERIMENT PREFIX
prefix = "testing new func"
# DIRECTORIES THAT NEED TO BE CONSIDERED
dir_names = [
    " (1)"
]
# IDENTIFIER TO PUT ON THE PLOT
identifier = "Just test"

# PLOT CONFIG
fig, axs = plt.subplots(5, 1, figsize=(16, 12))
fig.suptitle(f"{prefix} {identifier}", fontsize=20)
line_styles = ['-', '--', '-.', ':',(0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 2))]


for j, dir_name in enumerate(dir_names):
    # LOAD RESULTS
    # print(os.listdir(f"results"))
    with open(f"results/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
        metrics_results = pickle.load(file)

    # Remove the metrics that you do not want to include
    excluded = ["R2"]
    for out in excluded:
        metrics_results.pop(out)

    for i, metric in enumerate(metrics_results.keys()):
            
        y = metrics_results[metric]
        if metric == "Prediction Uncertainty":
            print(y)
        x = np.arange(0, len(y))
        axs[i].plot(x, y, label=dir_name, alpha=0.5,
                    linestyle=line_styles[j], linewidth=2)
        axs[i].set_title(metric)
        # PLOT LEGEND ONLY FOR ONE OF THEM
        if i == 2:
            axs[i].legend(loc='upper left', fontsize=15)

plt.tight_layout()
plt.savefig(f"{prefix} {identifier}")
plt.show()
plt.close()
