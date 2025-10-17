import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold


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

class FusionClassifier(nn.Module):
    def __init__(self, num_points, input_demo_dim, latent_shape=32):
        super().__init__()
        # Lightweight point cloud encoder
        self.sh_enc = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU()
        )
        self.fc_shape = nn.Sequential(
            nn.Linear(128, latent_shape),
            nn.ReLU()
        )
        # Joint classifier (uses demographics directly)
        self.joint = nn.Sequential(
            nn.Linear(latent_shape + input_demo_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, shape_x, demo_x):
        # shape_x: (B, N, 3)
        x = shape_x.transpose(1, 2)  # (B, 3, N)
        x = self.sh_enc(x)
        x = torch.max(x, 2)[0]  # (B, 128)
        z1 = self.fc_shape(x)
        z = torch.cat([z1, demo_x], dim=1)  # concat demographics directly
        out = self.joint(z)
        return out.squeeze(1)
    

# === Load demographic + label data ===
demo_df = pd.read_csv("/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/training_data/mixed_demographics.csv")
print(demo_df.head())


# Extract demographic numeric features
demo_df['sex'] = demo_df['sex'].astype(int)
demo_features = ['age', 'BMI', 'height', 'weight', 'diastolic_BP', 'systolic_BP', 'sex']
demo_df['height'] = demo_df['height']/100
X_demo = demo_df[demo_features].values
#scaler = StandardScaler()
#X_demo = scaler.fit_transform(X_demo)

# Encode labels (healthy=0, MI=1)
# Convert to 1 for MI, 0 for healthy
y = demo_df['MI'].map({'pMI': 1, 'healthy': 0}).values # or adjust to your labeling scheme
print(np.bincount(y))


# === Load and flatten all frames before PCA ===
shape_dir = "/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/training_data/mixed_samples"
files = sorted(glob.glob(os.path.join(shape_dir, "*.npy")))

heart_samples = []
for f in files:
    data = np.load(f)  # shape (10, 18000, 4)
    heart_samples.append(data)
heart_samples = np.array(heart_samples)  # (900, 10, 18000, 4)
print("Heart samples shape:", heart_samples.shape)

# Extract xyz only
heart_xyz = heart_samples[..., :3]  # (900, 10, 18000, 3)

# Flatten all frames per sample → (900, 540000)
X_flat = heart_xyz.reshape(len(heart_xyz), -1)
print("Flattened shape:", X_flat.shape)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=50)
X_shape = pca.fit_transform(X_flat)
print("PCA explained variance (first 10):", pca.explained_variance_ratio_[:10])
print("Reduced shape features:", X_shape.shape)  # shape (900, 2000, 3)

shape_dir = "/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/training_data/mixed_samples"
shape_files = sorted(glob.glob(os.path.join(shape_dir, "*.npy")))
# --- Load all frames per subject and flatten across frames ---
X_shape_points = []
for f in shape_files:
    arr = np.load(f)  # (10, 18000, 4)
    xyz = arr[..., :3]  # extract xyz
    flattened = xyz.reshape(-1, 3)  # combine all 10 frames -> (10*18000, 3)
    # Optional downsampling for speed
    if flattened.shape[0] > 10000:
        idx = np.random.choice(flattened.shape[0], 10000, replace=False)
        flattened = flattened[idx]
    X_shape_points.append(flattened)
X_shape_points = np.array(X_shape_points)
print("Combined point cloud shape (all frames):", X_shape_points.shape)

# Use a single train/test split (by indices) so both models see the same samples
n_samples = len(y)
indices = np.arange(n_samples)
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
print("Train size:", len(train_idx), "Test size:", len(test_idx))

# Split data for both pipelines
X_shape_points_train = X_shape_points[train_idx]
X_shape_points_test = X_shape_points[test_idx]
X_shape_pca_train = X_shape[train_idx]
X_shape_pca_test = X_shape[test_idx]
X_demo_train = X_demo[train_idx]
X_demo_test = X_demo[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(np.unique(y_train))


# =========================
# 1. Point cloud + demo model: FusionClassifier
# =========================
print("\n=== Training FusionClassifier (point cloud + demo) ===")
train_dataset = HeartDemoDataset(X_shape_points_train, X_demo_train, y_train)
test_dataset = HeartDemoDataset(X_shape_points_test, X_demo_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_points = X_shape_points.shape[1]
input_demo_dim = X_demo.shape[1]
fusion_model = FusionClassifier(num_points=num_points, input_demo_dim=input_demo_dim, latent_shape=32)
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=5e-6, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(10):
    fusion_model.train()
    for sx, dx, yy in train_loader:
        pred = fusion_model(sx.float(), dx.float())
        loss = criterion(pred, yy.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch", epoch, "loss", loss.item())

# --- Evaluation for FusionClassifier ---
fusion_model.eval()
all_preds_fusion, all_true = [], []
with torch.no_grad():
    for sx, dx, yy in test_loader:
        logits = fusion_model(sx.float(), dx.float())
        preds = torch.sigmoid(logits).numpy()
        all_preds_fusion.extend(preds)
        all_true.extend(yy.numpy())

pred_labels_fusion = (np.array(all_preds_fusion) > 0.5).astype(int)
acc_fusion = accuracy_score(all_true, pred_labels_fusion)
auc_fusion = roc_auc_score(all_true, all_preds_fusion)
cm_fusion = confusion_matrix(all_true, pred_labels_fusion)

print("\n=== FusionClassifier Diagnostics ===")
print(f"Test Accuracy: {acc_fusion:.4f}")
print(f"Test AUC: {auc_fusion:.4f}")
print("Confusion matrix:\n", cm_fusion)


# =========================
# 2. PCA-based XGBoost model (PCA+demo)
# =========================

print("\n=== Training XGBoost (PCA+demo) ===")
# Combine PCA features and demographics
X_pca_train = X_shape_pca_train
X_pca_test = X_shape_pca_test
X_xgb_train = np.concatenate([X_pca_train, X_demo_train], axis=1)
X_xgb_test = np.concatenate([X_pca_test, X_demo_test], axis=1)

# Initialize and train the XGBoost classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist",
    random_state=42
)

# --- Grid Search for XGBoost ---
from sklearn.model_selection import GridSearchCV

print("\n=== XGBoost Grid Search (AUC Optimization) ===")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_xgb_train, y_train)

print(f"Best AUC from Grid Search: {grid_search.best_score_:.4f}")
print("Best Parameters:", grid_search.best_params_)

# Use the best estimator from the grid search
xgb_model = grid_search.best_estimator_

# --- Evaluation for XGBoost ---
y_pred_prob_xgb = xgb_model.predict_proba(X_xgb_test)[:, 1]
y_pred_xgb = (y_pred_prob_xgb > 0.5).astype(int)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_prob_xgb)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

print("\n=== XGBoost Diagnostics ===")
print(f"Test Accuracy: {acc_xgb:.4f}")
print(f"Test AUC: {auc_xgb:.4f}")
print("Confusion matrix:\n", cm_xgb)


# =========================
# 3. Late-fusion Ensemble
# =========================
print("\n=== Late-fusion Ensemble ===")
# Use same test set for both models (all_true == y_test)
assert np.allclose(all_true, y_test)

# Weighted average ensemble (equal weights, or adjust as desired)
ensemble_preds = 0.5 * np.array(all_preds_fusion) + 0.5 * np.array(y_pred_prob_xgb)
ensemble_labels = (ensemble_preds > 0.5).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_labels)
ensemble_auc = roc_auc_score(y_test, ensemble_preds)
ensemble_cm = confusion_matrix(y_test, ensemble_labels)

print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
print(f"Ensemble Test AUC: {ensemble_auc:.4f}")
print("Ensemble confusion matrix:\n", ensemble_cm)
