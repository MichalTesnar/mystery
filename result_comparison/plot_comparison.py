import matplotlib.pyplot as plt
import pickle
import numpy as np

"""
Plot your metrics.
"""

prefix = "dagon hyperparams "
dir_names = [
#     "FIFO (0)",
#     "FIRO (0)",
#     "RIRO 0.3 (0)",
#     "RIRO 0.7 (0)",
#     "GREEDY (0)",
#     "THRESHOLD 0.3 (0)",
     "THRESHOLD 0.05 (0)",
#      "THRESHOLD 0.03 (0)",
     "THRESHOLD 0.01 (0)",
#     "THRESHOLD_GREEDY 0.3 (0)"
         "THRESHOLD_GREEDY 0.05 (0)",
        #  "THRESHOLD_GREEDY 0.03 (0)",
         "THRESHOLD_GREEDY 0.01 (0)",
]
prefix = "sinus hyperparams "
dir_names = [
     "FIFO (0)",
     "FIRO (0)",
     "RIRO 0.3 (0)",
     "GREEDY (0)",
     "THRESHOLD 0.3 (0)",
     "THRESHOLD_GREEDY 0.3 (0)",
             ]


identifier = "all"
fig, axs = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle(f"{prefix} {identifier}", fontsize=20)
line_styles = ['-', '--', '-.', ':',
               (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 2))]


for j, dir_name in enumerate(dir_names):
    with open(f"../results/sinus/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
        metrics_results = pickle.load(file)

    metrics_results.pop("R2") 

    for i, metric in enumerate(metrics_results.keys()):
        
        y = metrics_results[metric]
        x = np.arange(0, len(y))
        axs[i].plot(x, y, label=dir_name, alpha=0.5, linestyle=line_styles[j], linewidth=2)
        axs[i].set_title(metric)
        if i == 2:
                axs[i].legend(loc='upper left', fontsize=15)

plt.tight_layout()
plt.savefig(f"{prefix} {identifier}")
plt.show()
plt.close()
