import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# 1. Dataset wrapping
class HeartDemoDataset(Dataset):
    def __init__(self, shape_data, demo_data, labels):
        self.shape = shape_data
        self.demo = demo_data
        self.y = labels
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.shape[i], self.demo[i], self.y[i]

# 2. Encoders
class ShapeEncoder(nn.Module):
    """PointNet-style encoder for point cloud data"""
    def __init__(self, input_dim=3, latent_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        # x: (batch, num_points, 3)
        x = x.transpose(1, 2)  # -> (batch, 3, num_points)
        features = self.mlp(x)  # (batch, 256, num_points)
        global_feat = torch.max(features, 2)[0]  # symmetric function (max pooling)
        latent = self.fc(global_feat)
        return latent

class DemoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    def forward(self, d):
        return self.fc(d)

class FusionClassifier(nn.Module):
    def __init__(self, num_points, input_demo_dim, latent_shape=64, latent_demo=32):
        super().__init__()
        self.sh_enc = ShapeEncoder(input_dim=3, latent_dim=latent_shape)
        self.dm_enc = DemoEncoder(input_dim=input_demo_dim, latent_dim=latent_demo)
        self.joint = nn.Sequential(
            nn.Linear(latent_shape + latent_demo, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, shape_x, demo_x):
        z1 = self.sh_enc(shape_x)
        z2 = self.dm_enc(demo_x)
        z = torch.cat([z1, z2], dim=1)
        out = self.joint(z)
        return out.squeeze(1)  # logits
    

# === Load demographic + label data ===
demo_df = pd.read_csv("/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/training_data/mixed_demographics.csv")
print(demo_df.head())

# Assume the demographics file has columns:
# ['filename', 'age', 'BMI', 'height', 'weight', 'diastolic_BP', 'systolic_BP', 'sex', 'MI']
# Modify if different.

# Extract demographic numeric features
demo_features = ['age', 'BMI', 'height', 'weight', 'diastolic_BP', 'systolic_BP']
X_demo = demo_df[demo_features].values
scaler = StandardScaler()
X_demo = scaler.fit_transform(X_demo)

# Encode labels (healthy=0, MI=1)
# Convert to 1 for MI, 0 for healthy
y = demo_df['MI'].map({'pMI': 1, 'healthy': 0}).values # or adjust to your labeling scheme
print(np.bincount(y))

# === Load heart shape data ===
shape_dir = "/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/training_data/mixed_samples"
shape_files = sorted(os.listdir(shape_dir))

X_shape = []
for f in shape_files:
    path = os.path.join(shape_dir, f)
    arr = np.load(path)          # shape (10, 18000, 4)
    # frame0 = arr[0, :, :3]       # use first frame or average across frames
    frame0 = arr[0, np.random.choice(18000, 2000, replace=False), :3]
    X_shape.append(frame0)
X_shape = np.array(X_shape)      # shape (N, 18000, 3)

print("Shapes:")
print("X_shape:", X_shape.shape)
print("X_demo:", X_demo.shape)
print("y:", y.shape)

 # === Split into train/test sets ===
X_shape_train, X_shape_test, X_demo_train, X_demo_test, y_train, y_test = train_test_split(
    X_shape, X_demo, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", len(y_train), "Test size:", len(y_test))

# 3. Training and evaluation
train_dataset = HeartDemoDataset(X_shape_train, X_demo_train, y_train)
test_dataset = HeartDemoDataset(X_shape_test, X_demo_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_points = X_shape.shape[1]
input_demo_dim = X_demo.shape[1]
model = FusionClassifier(num_points=num_points, input_demo_dim=input_demo_dim, latent_shape=64, latent_demo=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(20):
    model.train()
    for sx, dx, yy in train_loader:
        pred = model(sx.float(), dx.float())
        loss = criterion(pred, yy.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch", epoch, "loss", loss.item())

# --- Evaluation ---
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for sx, dx, yy in test_loader:
        logits = model(sx.float(), dx.float())
        preds = torch.sigmoid(logits).numpy()
        all_preds.extend(preds)
        all_true.extend(yy.numpy())

# Threshold at 0.5
pred_labels = (np.array(all_preds) > 0.5).astype(int)
acc = accuracy_score(all_true, pred_labels)
auc = roc_auc_score(all_true, all_preds)
cm = confusion_matrix(all_true, pred_labels)

print("\n=== Diagnostics ===")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")
print("Confusion matrix:\n", cm)
