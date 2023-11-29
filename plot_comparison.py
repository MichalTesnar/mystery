import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import sys

sys.path.append('..')

# EXPERIMENT PREFIX
prefix = "Dagon try "
# DIRECTORIES THAT NEED TO BE CONSIDERED
dir_names = [
    "FIFO tuned (0)",
    "FIRO tuned (0)",
    "RIRO tuned (0)",
    "THRESHOLD tuned (2)",
    "GREEDY tuned (0)",
    "THRESHOLD_GREEDY tuned (2)"
]
# IDENTIFIER TO PUT ON THE PLOT
identifier = "fixed data"
excluded = ["Running Mean R2", "Cummulative MSE"]
HOW_MANY = 6 - len(excluded)
# PLOT CONFIG
plot_name = "Sinus Tuned ALL No FIFO"
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
    # print(os.listdir(f"results"))
    with open(f"results/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
        metrics_results = pickle.load(file)

    # Remove the metrics that you do not want to include
    
    # excluded = []
    for out in excluded:
        metrics_results.pop(out)

    for i, metric in enumerate(metrics_results.keys()):
            
        y = metrics_results[metric] 
        x = np.arange(0, len(y))
        axs[i].plot(x, y, label=dir_name, alpha=0.5,
                    linestyle=line_style(dir_name), linewidth=2, color=line_color(dir_name))
        # axs[i].set_title(metric)
        # PLOT LEGEND ONLY FOR ONE OF THEM
        axs[i].set_ylabel(metric)
    axs[1].legend(loc='center left', fontsize=12)
    axs[HOW_MANY-1].set_xlabel('upper left')

plt.tight_layout()
plt.savefig(f"{plot_name}")
plt.show()
plt.close()
