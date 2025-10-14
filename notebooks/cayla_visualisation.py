import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Get folder of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path relative to script
data_path = os.path.join(script_dir, "..", "data", "training_data", "mixed_samples", "0healthy.npy")

# Load data
data = np.load(data_path)
print("Data shape:", data.shape)
# Data shape: (10, 18000, 4)
# 10 samples
# 18000 points per sample
# 4 features at each point

num_samples = data.shape[0]
num_features = data.shape[2]
# Visualise all 10 samples
# plt.figure(figsize=(15, 20))  # Adjust height for clarity

# for s in range(num_samples):
#     plt.subplot(num_samples, 1, s + 1)
#     for f in range(num_features):
#         plt.plot(data[s, :, f], label=f"Feature {f+1}")
#     plt.title(f"Sample {s}")
#     if s == 0:
#         plt.legend(loc="upper right")
#     plt.ylabel("Value")
# plt.xlabel("Points along cardiac cycle")
# plt.tight_layout()
# plt.show()

# Load your data (already done)
# data.shape: (10, 18000, 4)

sample_idx = 0  # first sample
sample = data[sample_idx]

# Extract x, y, z coordinates
x = sample[:, 0]
y = sample[:, 1]
z = sample[:, 2]

# Plot 3D scatter
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', s=1, alpha=0.6)  # s=marker size, alpha=transparency

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"3D Shape of Heart for Sample {sample_idx}")
plt.show()

######
# Load tabular data
# Path to the CSV
csv_path = os.path.join(script_dir, "..", "data", "training_data", "mixed_demographics.csv")

# Load CSV
demographics = pd.read_csv(csv_path)

# Inspect the data
print("Demographics shape:", demographics.shape)
print(demographics.head())
print(demographics.info())

# Numeric features to visualise
numeric_features = ['age', 'BMI', 'height', 'weight', 'diastolic_BP', 'systolic_BP']
group_col = 'MI'  # healthy vs MI

# Boxplots
for feature in numeric_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=group_col, y=feature, data=demographics)
    plt.title(f"Boxplot of {feature} by MI status")
    plt.show()

# Violin plots
for feature in numeric_features:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x=group_col, y=feature, data=demographics)
    plt.title(f"Violin plot of {feature} by MI status")
    plt.show()

# Optional: plot sex distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', hue=group_col, data=demographics)
plt.title("Sex distribution by MI status")
plt.show()

# Pairwise scatterplot
# Include MI for hue
sns.pairplot(demographics[numeric_features + ['MI']], hue='MI', corner=True, 
             plot_kws={'alpha':0.6, 's':40})  # corner=True avoids duplicate plots

plt.suptitle("Pairwise Scatterplots of Demographic Features by MI Status", y=1.02)
plt.show()
