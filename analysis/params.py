import matplotlib.pyplot as plt
import pickle
import numpy as np

def line_style(st):
    return '-'


def line_color(st):
    if "0.0016" in st or "0.1" in st or " 10" in st:
        return colors[0]
    elif "0.0023" in st or "0.2" in st or " 25" in st:
        return colors[1]
    elif "0.0036" in st or "0.3" in st or " 50" in st:
        return colors[2]
    elif "0.0056" in st or "0.4" in st or " 100" in st:
        return colors[3]
    elif "0.0075" in st or "0.5" in st or " 200" in st:
        return colors[4]
    elif "0.0096" in st or "0.6" in st or " 400" in st:
        return colors[5]
    elif "0.012" in st or "0.7" in st:
        return colors[6]
    elif "0.0156" in st or "0.8" in st:
        return colors[7]
    elif "0.0228" in st or "0.9" in st:
        return colors[8]
    
def extracted_name(st):
    st = st.replace("tuned (0)", "")
    st = st.replace("THRESHOLD_GREEDY", "Threshold Greedy")
    st = st.replace("THRESHOLD", "Threshold")
    st = st.replace("BUFFER", "")
    return st
    

# EXPERIMENT PREFIX
prefix = "Full data fix "
# DIRECTORIES THAT NEED TO BE CONSIDERED
plot_name = "RIRO"

if "BUFFER Threshold Greedy" in plot_name:
    dir_names = [
    "THRESHOLD_GREEDY 10 BUFFER tuned (0)",
    "THRESHOLD_GREEDY 25 BUFFER tuned (0)",
    "THRESHOLD_GREEDY 50 BUFFER tuned (0)",
    "THRESHOLD_GREEDY 100 BUFFER tuned (0)",
    "THRESHOLD_GREEDY 200 BUFFER tuned (0)",
    "THRESHOLD_GREEDY 400 BUFFER tuned (0)"
    ]
elif "BUFFER RIRO" in plot_name:
    dir_names = [
    "RIRO 10 BUFFER tuned (0)",
    "RIRO 25 BUFFER tuned (0)",
    "RIRO 50 BUFFER tuned (0)",
    "RIRO 100 BUFFER tuned (0)",
    "RIRO 200 BUFFER tuned (0)",
    "RIRO 400 BUFFER tuned (0)"
    ]
elif "BUFFER FIRO" in plot_name:
    dir_names = [
    "FIRO 10 BUFFER tuned (0)",
    "FIRO 25 BUFFER tuned (0)",
    "FIRO 50 BUFFER tuned (0)",
    "FIRO 100 BUFFER tuned (0)",
    "FIRO 200 BUFFER tuned (0)",
    "FIRO 400 BUFFER tuned (0)"
    ]
elif "BUFFER Greedy" in plot_name:
    dir_names = [
    "GREEDY 10 BUFFER tuned (2)",
    "GREEDY 25 BUFFER tuned (2)",
    "GREEDY 50 BUFFER tuned (2)",
    "GREEDY 100 BUFFER tuned (2)",
    "GREEDY 200 BUFFER tuned (2)",
    "GREEDY 400 BUFFER tuned (2)"
    ]
elif "BUFFER Threshold" in plot_name:
    dir_names = [
    "THRESHOLD 10 BUFFER tuned (0)",
    "THRESHOLD 25 BUFFER tuned (0)",
    "THRESHOLD 50 BUFFER tuned (0)",
    "THRESHOLD 100 BUFFER tuned (0)",
    "THRESHOLD 200 BUFFER tuned (0)",
    "THRESHOLD 400 BUFFER tuned (0)"
    ]

elif "BUFFER FIFO" in plot_name:
    dir_names = [
    "FIFO 10 BUFFER tuned (0)",
    "FIFO 25 BUFFER tuned (0)",
    "FIFO 50 BUFFER tuned (0)",
    "FIFO 100 BUFFER tuned (0)",
    "FIFO 200 BUFFER tuned (0)",
    "FIFO 400 BUFFER tuned (0)"
    ]
elif "Threshold Greedy" in plot_name:
    dir_names = [
    "THRESHOLD_GREEDY 0.0016 tuned (0)",
    "THRESHOLD_GREEDY 0.0023 tuned (0)",
    "THRESHOLD_GREEDY 0.0036 tuned (0)",
    "THRESHOLD_GREEDY 0.0056 tuned (0)",
    "THRESHOLD_GREEDY 0.0075 tuned (0)",
    "THRESHOLD_GREEDY 0.0096 tuned (0)",
    "THRESHOLD_GREEDY 0.012 tuned (0)",
    "THRESHOLD_GREEDY 0.0156 tuned (0)",
    "THRESHOLD_GREEDY 0.0228 tuned (0)"
        ]
elif "Threshold" in plot_name:
    dir_names = [
    "THRESHOLD 0.0016 tuned (0)",
    "THRESHOLD 0.0023 tuned (0)",
    "THRESHOLD 0.0036 tuned (0)",
    "THRESHOLD 0.0056 tuned (0)",
    "THRESHOLD 0.0075 tuned (0)",
    "THRESHOLD 0.0096 tuned (0)",
    "THRESHOLD 0.012 tuned (0)",
    "THRESHOLD 0.0156 tuned (0)",
    "THRESHOLD 0.0228 tuned (0)"
    ]
elif "RIRO" in plot_name:
    dir_names = [
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

colors = plt.get_cmap('gist_rainbow')(np.linspace(0, 1, len(dir_names)))
# IDENTIFIER TO PUT ON THE PLOT
excluded = {"MSE": 0,
            "R2": 1,
            "Cummulative MSE": 1, 
            "Prediction Uncertainty": 0,
            "Skips": 0,
            }
true_labels = [label for label, value in excluded.items() if value]
HOW_MANY = sum([1 if i else 0 for i in excluded.values()])
# PLOT CONFIG
plot_name = plot_name + "_" + "_".join(true_labels)

fig, axs = plt.subplots(HOW_MANY, 1, figsize=(16, 11), sharex=True)
FONT_SIZE = 20
FONT_SIZE_TICKS = 15

# fig.suptitle(f"{plot_name}", fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE_TICKS)
plt.yticks(fontsize=FONT_SIZE_TICKS)

try:
    len(axs)
except:
    axs = [axs]

y_max = -float("inf")

for j, dir_name in enumerate(dir_names):
    with open(f"results/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
        metrics_results = pickle.load(file)
    i = 0

    for metric in metrics_results.keys():
        

        if not excluded[metric]:
            continue

        y = metrics_results[metric]

        y_max = max(y_max, len(y))

        if len(y) < y_max:
            zeros_array = np.full((y_max - len(y),), np.nan) # print(np.zeros(, dtype=y.dtype).shape)
            y = np.concatenate((zeros_array, y))

        if metric == "Skips":
            print(dir_name, y[-1])
        x = np.arange(0, len(y))
        if metric == "MSE":
            y = np.minimum(1, y)
        if metric == "R2":
            y = np.maximum(-1.5, y)

        # axs[i].axhline(y=0.001, color=line_color("BASELINE"), label=extracted_name(dir_name))

        if "OFFLINE" in dir_name and metric in ["MSE", "R2"]:
            axs[i].axhline(y=y, color=line_color("BASELINE"), label=extracted_name(dir_name))
        else:
            axs[i].plot(x, y, label=extracted_name(dir_name), alpha=0.5,
                        linestyle=line_style(dir_name), linewidth=1.5, color=line_color(dir_name))
            axs[i].set_ylabel(metric, fontsize=FONT_SIZE)

        i += 1
    if excluded["R2"]:
        location = 'lower right'
    if excluded["MSE"] or excluded["Cummulative MSE"] or excluded["Skips"]:
        location = 'upper left'
    elif excluded["Prediction Uncertainty"]:
        location = 'upper right'
    axs[min(HOW_MANY-1, 2)].legend(loc=location, fontsize=FONT_SIZE)
    axs[HOW_MANY-1].set_xlabel('Iterations', fontsize=FONT_SIZE)

plt.tight_layout()
plt.savefig(f"{plot_name}")
plt.show()
plt.close()



