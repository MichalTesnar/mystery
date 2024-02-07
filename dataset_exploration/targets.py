import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
csv_file_path = "dagon_dataset.csv"
df = pd.read_csv(csv_file_path, skiprows=1, nrows=2000)
u_dot = df.iloc[:, 6]
v_dot = df.iloc[:, 7]
r_dot = df.iloc[:, 8]
x = np.arange(0, len(u_dot))

# Create a figure and three subplots arranged vertically
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True)
fig.suptitle('AUV Dagon Dataset Targets', fontsize=20)

# Plot data
ax1.plot(x, u_dot, '.', markersize=1, label=r'$\dot{u}$ $(m/s^2)$')
ax2.plot(x, v_dot, '.', markersize=1, label=r'$\dot{v}$ $(m/s^2)$')
ax3.plot(x, r_dot, '.', markersize=1, label=r'$\dot{r}$ $(rd/s^2)$')
# ax1.legend(loc='upper left', fontsize="20")
# ax2.legend(loc='upper left', fontsize="20")
# ax3.legend(loc='upper left', fontsize="20")
FONTSIZE = 15
ax3.set_xlabel(r'$t(s)$', fontsize=FONTSIZE)
ax1.set_ylabel(r'$\dot{u}$ $(m/s^2)$', fontsize=FONTSIZE)
ax2.set_ylabel(r'$\dot{v}$ $(m/s^2)$', fontsize=FONTSIZE)
ax3.set_ylabel(r'$\dot{r}$ $(rd/s^2)$', fontsize=FONTSIZE)

# Show the plots
plt.tight_layout()
plt.savefig("targets.pdf")
plt.show()
