import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Specify the path to your CSV file
csv_file_path = "dagon_dataset.csv"
df = pd.read_csv(csv_file_path, skiprows=1)
th1 = df.iloc[:, 3]
th2 = df.iloc[:, 4]
th3 = df.iloc[:, 5]
x = np.arange(0, len(th1))

# Create a figure and three subplots arranged vertically
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle('Values of the Thrusters of AUV Dagon in the Dataset', fontsize=16)

# Plot data on the first subplot
ax1.plot(x, th1, label='Thruster 1')
ax1.set_title('Thruster 1')
ax1.legend()
ax2.plot(x, th2, label='Thruster 2')
ax2.set_title('Thruster 2')
ax2.legend()
ax3.plot(x, th3, label='Thruster 3')
ax3.set_title('Thruster 3')
ax3.legend()

# Show the plots
plt.tight_layout()
plt.show()
