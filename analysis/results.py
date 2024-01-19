import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np


def line_style(st):
    return '-'
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
        return (0, (1, 2))


def extracted_name(st):
    if "FIFO" in st:
        return "FIFO"
    elif "FIRO" in st:
        return "FIRO"
    elif "RIRO" in st:
        return 'RIRO'
    elif "THRESHOLD_GREEDY" in st:
        return "Threshold-Greedy"
    elif "THRESHOLD" in st:
        return "Threshold"
    elif "GREEDY" in st:
        return "Greedy"
    elif "OFFLINE" in st:  # Added condition for a new color
        return "Offline"


def line_color(st):
    if "FIFO" in st:
        return 'limegreen'
    elif "FIRO" in st:
        return 'blue'
    elif "RIRO" in st:
        return 'magenta'
    elif "THRESHOLD_GREEDY" in st:
        return 'darkorange'
    elif "THRESHOLD" in st:
        return 'gold'
    elif "GREEDY" in st:
        return 'red'
    elif "BASELINE" in st:  # Added condition for a new color
        return 'black'
    else:
        return 'pink'


# EXPERIMENT PREFIX
prefix = "Full data fix "
# DIRECTORIES THAT NEED TO BE CONSIDERED
# dir_names = [
#     "OFFLINE  tuned (0)",
#     "FIFO  tuned (0)",
#     "FIRO  tuned (0)",
#     "RIRO  tuned (0)",
#     "GREEDY  tuned (0)",
#     "THRESHOLD  tuned (0)",
#     "THRESHOLD_GREEDY  tuned (0)"
# ]

dir_names = [
    "OFFLINE  tuned (0)",
    "FIFO  tuned (0)",
    "FIRO  tuned (0)",
    "RIRO 0.2 tuned (0)",
    "GREEDY  tuned (0)",
    "THRESHOLD 0.0156 tuned (0)",
    "THRESHOLD_GREEDY 0.0228 tuned (0)"
]

# IDENTIFIER TO PUT ON THE PLOT
excluded = {"MSE": False,
            "R2": False,
            "Cummulative MSE": False,
            "Prediction Uncertainty": True,
            "Skips": False,
            }
true_labels = [label for label, value in excluded.items() if value]
HOW_MANY = sum([1 if i else 0 for i in excluded.values()])
# PLOT CONFIG
plot_name = "final_go_" + "_".join(true_labels)
fig, axs = plt.subplots(HOW_MANY, 1, figsize=(14, 11), sharex=True)
FONT_SIZE = 20
FONT_SIZE_TICKS = 15

# fig.suptitle(f"{plot_name}", fontsize=FONT_SIZE)
# convert axs to array

matplotlib.rc('xtick', labelsize=FONT_SIZE_TICKS) 
matplotlib.rc('ytick', labelsize=FONT_SIZE_TICKS) 

try:
    len(axs)
except:
    axs = [axs]

for ax in axs:
    ax.tick_params('y', labelsize=FONT_SIZE_TICKS) 
    ax.tick_params('x', labelsize=FONT_SIZE_TICKS) 

for j, dir_name in enumerate(dir_names):
    with open(f"results/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
        metrics_results = pickle.load(file)
    i = 0

    for metric in metrics_results.keys():
        if not excluded[metric]:
            print(dir_name, f"{len(metrics_results[metric])-metrics_results[metric][-1]}/{len(metrics_results[metric])}")
            continue

        y = metrics_results[metric]
        x = np.arange(0, len(y))
        if metric == "MSE":
            y = np.minimum(0.035, y)
        if metric == "Prediction Uncertainty":
            y = np.minimum(0.15, y)
        if metric == "R2":
            y = np.maximum(-1.5, y)

        if "OFFLINE" in dir_name:
            if metric in ["MSE", "R2"]:
                axs[i].axhline(y=y, color=line_color("BASELINE"), label=extracted_name(dir_name))
            else:
                axs[i].axhline(y=np.nan, color=line_color("BASELINE"), label=extracted_name(dir_name))
        else:
            axs[i].plot(x, y, label=extracted_name(dir_name), alpha=0.5,
                        linestyle=line_style(dir_name), linewidth=1.5, color=line_color(dir_name))
        axs[i].set_ylabel(metric, fontsize=FONT_SIZE)

        i += 1
    if excluded["R2"]:
        location = 'lower right'
    elif excluded["MSE"] or excluded["Cummulative MSE"] or excluded["Skips"]:
        location = 'upper left'
    elif excluded["Prediction Uncertainty"]:
        location = 'upper right'
    axs[min(HOW_MANY-1, 2)].legend(loc=location, fontsize=FONT_SIZE-3)
    axs[HOW_MANY-1].set_xlabel('Iterations', fontsize=FONT_SIZE)

plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
plt.tight_layout()
plt.savefig(f"{plot_name}.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()
