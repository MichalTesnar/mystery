import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import sys

sys.path.append('..')

# EXPERIMENT PREFIX
prefix = "sinus: first real try "
# DIRECTORIES THAT NEED TO BE CONSIDERED
dir_names = [
    # "FIFO tuned (0)",
    # "FIRO tuned (0)",
    "RIRO tuned (0)",
    "THRESHOLD tuned (0)",
    "GREEDY tuned (0)",
    "THRESHOLD_GREEDY tuned (0)"
]
# IDENTIFIER TO PUT ON THE PLOT
identifier = "fixed data"
excluded = {"MSE": True,
            "R2": True,
            "Running Mean R2": False,
            "Cummulative MSE": False, 
            "Prediction Uncertainty": False,
            "Skips": False,
            }
HOW_MANY = sum([1 if i else 0 for i in excluded.values()])
# PLOT CONFIG
plot_name = "Sinus Tuned Selection"
fig, axs = plt.subplots(HOW_MANY, 1, figsize=(16, 13), sharex=True)
fig.suptitle(f"{plot_name}", fontsize=15)
def line_style(st):
    if "FIFO" in st:
        return '-'
    elif "FIRO" in st:
        return '--'
    elif "RIRO" in st:
        return '-.'
    elif "THRESHOLD_GREEDY" in st:
        return ':'
    elif "THRESHOLD" in st:
        return (0, (3, 1, 1, 1))
    elif "GREEDY" in st:
        return (0, (5, 2))
    else:
        return(0, (1, 2))


def line_color(st):
    if "FIFO" in st:
        return 'green'
    elif "FIRO" in st:
        return 'blue'
    elif "RIRO" in st:
        return 'red'
    elif "THRESHOLD_GREEDY" in st:
        return 'brown'
    elif "THRESHOLD" in st:
        return 'black'
    elif "GREEDY" in st:
        return 'orange'
    else:
        return 'pink'

for j, dir_name in enumerate(dir_names):
    # LOAD RESULTS
    with open(f"results/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
        metrics_results = pickle.load(file)

    for i, metric in enumerate(metrics_results.keys()):
        if not excluded[metric]:
            continue
        y = metrics_results[metric] 
        x = np.arange(0, len(y))
        axs[i].plot(x, y, label=dir_name, alpha=0.5,
                    linestyle=line_style(dir_name), linewidth=2, color=line_color(dir_name))
        # axs[i].set_title(metric)
        # PLOT LEGEND ONLY FOR ONE OF THEM
        axs[i].set_ylabel(metric)
    axs[min(HOW_MANY-1, 2)].legend(loc='center left', fontsize=12)
    axs[HOW_MANY-1].set_xlabel('Iterations')

plt.tight_layout()
plt.savefig(f"{plot_name}")
plt.show()
plt.close()
