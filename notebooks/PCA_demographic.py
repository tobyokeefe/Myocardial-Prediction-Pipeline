import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Path to CSV file
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, "..", "data", "training_data", "mixed_demographics.csv")

# Load the CSV data
df = pd.read_csv(data_path)
print(df.head())

# Select numeric columns only (exclude MI and sex)
numeric_cols = ['age', 'BMI', 'height', 'weight', 'diastolic_BP', 'systolic_BP']
X = df[numeric_cols]

# Scatterplot
sns.pairplot(df, vars=['age', 'BMI', 'height', 'weight', 'diastolic_BP', 'systolic_BP'],
             hue='MI', diag_kind='kde')
plt.suptitle("Original Variables - Pairwise Relationships", y=1.02)
plt.show()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance (as percentages)
np.set_printoptions(precision=2, suppress=True)

explained_var = pca.explained_variance_ratio_ * 100
cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100

print("Explained variance ratio (%):", explained_var)
print("Cumulative variance (%):", cumulative_var)

# Scree plot
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Scree Plot")
plt.grid(True)
plt.show()

# Scatterplot 
# After PCA
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df['MI'] = df['MI']

sns.pairplot(
    pca_df,
    vars=[f'PC{i+1}' for i in range(6)],  # show all six PCs
    hue='MI',
    diag_kind='kde'
)
plt.suptitle("All Principal Components - Pairwise Relationships", y=1.02)
plt.show()

# Keep only first 3 PCs
# Scatterplot - After PCA (first 3 components only)
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df['MI'] = df['MI']

sns.pairplot(
    pca_df,
    vars=[f'PC{i+1}' for i in range(3)],  # only PC1, PC2, PC3
    hue='MI',
    diag_kind='kde'
)
plt.suptitle("First 3 Principal Components - Pairwise Relationships", y=1.02)
plt.show()

# Single visualisation
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=(df["MI"] == "pMI"), cmap='coolwarm', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PC1 vs PC2 (colored by MI)")
plt.show()