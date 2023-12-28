import matplotlib.pyplot as plt
import pickle
import numpy as np

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
    elif "BASELINE" in st:  # Added condition for a new color
        return 'purple'
    else:
        return 'pink'


# EXPERIMENT PREFIX
prefix = "Full data "
# DIRECTORIES THAT NEED TO BE CONSIDERED
dir_names = [
    # "OFFLINE tuned (0)",
    "FIFO tuned (0)",
    "FIRO tuned (0)",
    "RIRO tuned (0)",
    "GREEDY tuned (0)",
    "THRESHOLD tuned (0)",
    "THRESHOLD_GREEDY tuned (0)"
]
# IDENTIFIER TO PUT ON THE PLOT
excluded = {"MSE": True,
            "R2": True,
            "Running Mean R2": False,
            "Cummulative MSE": False, 
            "Prediction Uncertainty": True,
            "Skips": False,
            }
# R2_BASELINE = 0.8
MSE_BASELINE = 0.00697501
HOW_MANY = sum([1 if i else 0 for i in excluded.values()])
# PLOT CONFIG
plot_name = "Dagon No FIFO All Data Points Rest"
fig, axs = plt.subplots(HOW_MANY, 1, figsize=(16, 13), sharex=True)
fig.suptitle(f"{plot_name}", fontsize=15)


for j, dir_name in enumerate(dir_names):
    with open(f"results/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
        metrics_results = pickle.load(file)
    # if "OFFLINE" in dir_name:
    #     # R2_BASELINE = metrics_results["R2"]
    #     # print(R2_BASELINE)
    #     MSE_BASELINE = metrics_results["MSE"]
    #     print(MSE_BASELINE)
    #     continue
    i = 0
    
    try:
        len(axs)
    except:
        axs = [axs]

    for metric in metrics_results.keys():
        if not excluded[metric]:
            continue
        y = metrics_results[metric]
        x = np.arange(0, len(y))
        axs[i].plot(x, y, label=dir_name, alpha=0.5,
                    linestyle=line_style(dir_name), linewidth=2, color=line_color(dir_name))
        axs[i].set_ylabel(metric)
        if metric == "MSE":
            axs[i].axhline(y=MSE_BASELINE, color=line_color("BASELINE"), label="Baseline")
        # if metric == "R2":
        #     axs[i].axhline(y=R2_BASELINE, color=line_color("BASELINE"), label="Baseline")
        i += 1
    axs[min(HOW_MANY-1, 2)].legend(loc='center left', fontsize=12)
    axs[HOW_MANY-1].set_xlabel('Iterations')

plt.tight_layout()
# plt.savefig(f"{plot_name}")
plt.show()
plt.close()
