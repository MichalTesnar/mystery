import matplotlib.pyplot as plt
import pickle
import numpy as np
import re


def line_style(st):
    if "FIFO" in st or "FIRO" in st or "RIRO" in st or "OFFLINE" in st:
        return '--'
    else:
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
    
def line_color2(st):
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


def extracted_name(st):
    st = st.replace("tuned (0)", "")
    st = st.replace("THRESHOLD_GREEDY", "Threshold Greedy")
    st = st.replace("THRESHOLD", "Threshold")
    st = st.replace("BUFFER", "")
    st = st.replace("GREEDY", "Greedy")
    return st


# EXPERIMENT PREFIX
prefix = "Full data fix "
# DIRECTORIES THAT NEED TO BE CONSIDERED

BUFFER_Threshold_Greedy = [
    "THRESHOLD_GREEDY 10 BUFFER tuned (0)",
    "THRESHOLD_GREEDY 25 BUFFER tuned (0)",
    "THRESHOLD_GREEDY 50 BUFFER tuned (0)",
    "THRESHOLD_GREEDY 100 BUFFER tuned (0)",
    "THRESHOLD_GREEDY 200 BUFFER tuned (0)",
    "THRESHOLD_GREEDY 400 BUFFER tuned (0)"
]
BUFFER_RIRO = [
    "RIRO 10 BUFFER tuned (0)",
    "RIRO 25 BUFFER tuned (0)",
    "RIRO 50 BUFFER tuned (0)",
    "RIRO 100 BUFFER tuned (0)",
    "RIRO 200 BUFFER tuned (0)",
    "RIRO 400 BUFFER tuned (0)"
]
BUFFER_FIRO = [
    "FIRO 10 BUFFER tuned (0)",
    "FIRO 25 BUFFER tuned (0)",
    "FIRO 50 BUFFER tuned (0)",
    "FIRO 100 BUFFER tuned (0)",
    "FIRO 200 BUFFER tuned (0)",
    "FIRO 400 BUFFER tuned (0)"
]
BUFFER_Greedy = [
    "GREEDY 10 BUFFER tuned (2)",
    "GREEDY 25 BUFFER tuned (2)",
    "GREEDY 50 BUFFER tuned (2)",
    "GREEDY 100 BUFFER tuned (2)",
    "GREEDY 200 BUFFER tuned (2)",
    "GREEDY 400 BUFFER tuned (2)"
]
BUFFER_Threshold = [
    "THRESHOLD 10 BUFFER tuned (0)",
    "THRESHOLD 25 BUFFER tuned (0)",
    "THRESHOLD 50 BUFFER tuned (0)",
    "THRESHOLD 100 BUFFER tuned (0)",
    "THRESHOLD 200 BUFFER tuned (0)",
    "THRESHOLD 400 BUFFER tuned (0)"
]

BUFFER_FIFO = [
    "FIFO 10 BUFFER tuned (0)",
    "FIFO 25 BUFFER tuned (0)",
    "FIFO 50 BUFFER tuned (0)",
    "FIFO 100 BUFFER tuned (0)",
    "FIFO 200 BUFFER tuned (0)",
    "FIFO 400 BUFFER tuned (0)"
]
Threshold_Greedy = [
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
Threshold = [
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
RIRO = [
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


def find_number(text):
    # Updated pattern to allow integers and decimal numbers
    pattern = r'\b(\d+\.\d+|\d+)\b'

    matches = re.findall(pattern, text)

    if matches:
        # Filter out numbers within brackets
        filtered_numbers = [float(match)
                            for match in matches if '(' not in match]

        if filtered_numbers:
            return filtered_numbers[0]
        else:
            raise ValueError("No valid number found in the given string.")
    else:
        raise ValueError("No number found in the given string.")


# PLOT CONFIG


fig, axs = plt.subplots(1, 1, figsize=(16, 11), sharex=True)
FONT_SIZE = 30
FONT_SIZE_TICKS = 30

# fig.suptitle(f"{plot_name}", fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE_TICKS)
plt.yticks(fontsize=FONT_SIZE_TICKS)

try:
    len(axs)
except:
    axs = [axs]

dirs_RIRO = ([RIRO], 'RIRO CMSE', r'Values of $P$ in RIRO')

dirs_Threshold = ([Threshold], 'Threshold CMSE', r'Values of $t$ in Threshold')

dirs_ThresholdGreedy = ([Threshold_Greedy], 'Threshold-Greedy CMSE', r'Values of $t$ in Threshold-Greedy')

dirs_BUFFER = ([
    BUFFER_FIFO,
    BUFFER_FIRO,
    BUFFER_RIRO,
    BUFFER_Greedy,
    BUFFER_Threshold,
    BUFFER_Threshold_Greedy
], 'BUFFER CMSE', r'Buffer Sizes')


dirs_to_handle, plot_name, plot_string = dirs_BUFFER

colors = plt.get_cmap('gist_rainbow')(
    np.linspace(0, 1, len(dirs_to_handle[0])))


for dir_names in dirs_to_handle:
    params = []
    resulting_CMSE = []

    for j, dir_name in enumerate(dir_names):
        with open(f"results/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
            metrics_results = pickle.load(file)

        y = metrics_results["Cummulative MSE"][-1]
        resulting_CMSE.append(y)
        x = find_number(dir_name)
        params.append(x)

    axs[0].plot(params, resulting_CMSE, marker='o',
                linestyle=line_style(dir_name), color=line_color2(dir_name), label=extracted_name(str(dir_name.split(" ")[0])))

    axs[0].set_xlabel(plot_string, fontsize=FONT_SIZE)
    axs[0].set_ylabel("Cumulative MSE", fontsize=FONT_SIZE)

    if plot_name == 'BUFFER CMSE':
        axs[0].legend(fontsize=FONT_SIZE)

plt.tight_layout()
plt.savefig(f"{plot_name}.pdf", format="pdf", bbox_inches="tight")

plt.show()
plt.close()
