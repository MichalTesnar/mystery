import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Specify the path to your CSV file
csv_file_path = "dagon_dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, skiprows=1)
# Extract specific columns by column name
x = df.iloc[:, 0]
y = df.iloc[:, 1]
z = df.iloc[:, 2]

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cmap = plt.get_cmap('viridis')
colors = np.arange(len(x))
normalize = plt.Normalize(colors.min(), colors.max())
colormap = cmap(normalize(colors))

# Scatter plot the points
ax.scatter(x, y, z, c=colormap, cmap='viridis', marker='o')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
