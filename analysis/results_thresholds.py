import matplotlib.pyplot as plt
import pickle
import numpy as np

def line_style(st):
    # if "0.0016" in st:
        return '-'
    # elif "0.0023" in st:
    #     return '--'
    # elif "0.0036" in st:
    #     return '-.'
    # elif "0.0056" in st:
    #     return ':'
    # elif "0.0075" in st:
    #     return (0, (3, 1, 1, 1))
    # elif "0.0096" in st:
    #     return (0, (5, 2))
    # else:
    #     return(0, (1, 2))

def line_color(st):
    if "0.0016" in st:
        return 'green'
    elif "0.0023" in st:
        return 'blue'
    elif "0.0036" in st:
        return 'red'
    elif "0.0056" in st:
        return 'brown'
    elif "0.0075" in st:
        return 'black'
    elif "0.0096" in st:
        return 'orange'
    elif "0.012" in st:  # Added condition for a new color
        return 'purple'
    elif "0.0156" in st:  # Added condition for a new color
        return 'pink'
    elif "0.0228" in st:  # Added condition for a new color
        return 'cyan'
    

# EXPERIMENT PREFIX
prefix = "Full data "
# DIRECTORIES THAT NEED TO BE CONSIDERED
dir_names = [
    # "OFFLINE tuned (0)",
    # "FIFO tuned (0)",
    # "FIRO tuned (0)",
    # "RIRO tuned (0)",
    # "GREEDY tuned (0)",
    # "THRESHOLD tuned (0)",
    # "THRESHOLD_GREEDY tuned (0)",
# "THRESHOLD_GREEDY 0.0016 tuned (0)",
# "THRESHOLD_GREEDY 0.0023 tuned (0)",
# "THRESHOLD_GREEDY 0.0036 tuned (0)",
# "THRESHOLD_GREEDY 0.0056 tuned (0)",
# "THRESHOLD_GREEDY 0.0075 tuned (0)",
# "THRESHOLD_GREEDY 0.0096 tuned (0)",
# "THRESHOLD_GREEDY 0.012 tuned (0)",
# "THRESHOLD_GREEDY 0.0156 tuned (0)",
# "THRESHOLD_GREEDY 0.0228 tuned (0)"
"RIRO 0.1 tuned (0)",
"RIRO 0.2 tuned (0)",
"RIRO 0.3 tuned (0)",
"RIRO 0.4 tuned (0)",
"RIRO 0.5 tuned (0)",
"RIRO 0.6 tuned (0)",
"RIRO 0.7 tuned (0)",
"RIRO 0.8 tuned (0)",
"RIRO 0.9 tuned (0)",
]
# IDENTIFIER TO PUT ON THE PLOT
excluded = {"MSE": False,
            "R2": False,
            "Running Mean R2": True,
            "Cummulative MSE": True, 
            "Prediction Uncertainty": False,
            "Skips": True,
            }
# R2_BASELINE = 0.8
# MSE_BASELINE = 0.002
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
        # if metric == "MSE":
        #     axs[i].axhline(y=MSE_BASELINE, color=line_color("BASELINE"), label="Baseline")
        # if metric == "R2":
        #     axs[i].axhline(y=R2_BASELINE, color=line_color("BASELINE"), label="Baseline")
        i += 1
    axs[min(HOW_MANY-1, 2)].legend(loc='center left', fontsize=12)
    axs[HOW_MANY-1].set_xlabel('Iterations')

plt.tight_layout()
# plt.savefig(f"{plot_name}")
plt.show()
plt.close()
