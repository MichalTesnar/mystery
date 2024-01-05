import matplotlib.pyplot as plt
import pickle
import numpy as np

def line_style(st):
    return '-'

colors = plt.get_cmap('viridis')(np.linspace(0, 1, 9))
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
    return st
    

# EXPERIMENT PREFIX
prefix = "Full data fix "
# DIRECTORIES THAT NEED TO BE CONSIDERED
plot_name = "Threshold Greedy"

if "Threshold Greedy" in plot_name:
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
elif "Buffer" in plot_name:
    dir_names = [
    "TBA"
    ]
# IDENTIFIER TO PUT ON THE PLOT
excluded = {"MSE": False,
            "R2": False,
            "Cummulative MSE": True, 
            "Prediction Uncertainty": False,
            "Skips": False,
            }
HOW_MANY = sum([1 if i else 0 for i in excluded.values()])
# PLOT CONFIG

fig, axs = plt.subplots(HOW_MANY, 1, figsize=(16, 11), sharex=True)
FONT_SIZE = 20
FONT_SIZE_TICKS = 15

fig.suptitle(f"{plot_name}", fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE_TICKS)
plt.yticks(fontsize=FONT_SIZE_TICKS)

try:
    len(axs)
except:
    axs = [axs]

for j, dir_name in enumerate(dir_names):
    with open(f"results/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
        metrics_results = pickle.load(file)
    i = 0

    for metric in metrics_results.keys():
        

        if not excluded[metric]:
            continue

        y = metrics_results[metric]
        if metric == "Skips":
            print(dir_name, y[-1])
        x = np.arange(0, len(y))
        if metric == "R2":
            y = np.maximum(-.5, y)

        if "OFFLINE" in dir_name and metric in ["MSE", "R2"]:
            axs[i].axhline(y=y, color=line_color("BASELINE"), label=extracted_name(dir_name))
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
    axs[min(HOW_MANY-1, 2)].legend(loc=location, fontsize=FONT_SIZE)
    axs[HOW_MANY-1].set_xlabel('Iterations', fontsize=FONT_SIZE)

plt.tight_layout()
# plt.savefig(f"{plot_name}")
plt.show()
plt.close()



