import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
heart_flat = heart_xyz.reshape(900, -1)   # (900, 10*18000*3) = (900, 540000)

demo_df = pd.read_csv("/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/training_data/mixed_demographics.csv")
# Include 'sex' column as numeric (convert True/False to 1/0)
demo_df['sex'] = demo_df['sex'].astype(int)
demo_features = ['age', 'BMI', 'height', 'weight', 'diastolic_BP', 'systolic_BP', 'sex']
print("Included 'sex' as a demographic feature (0=Male, 1=Female)")
demo_df['height'] = demo_df['height'] / 100 
X_demo = demo_df[demo_features].values
y = demo_df['MI'].map({'pMI': 1, 'healthy': 0}).values # or adjust to your labeling scheme



X_combined = np.hstack([heart_flat, X_demo])  
print("Combined feature shape:", X_combined.shape)  


X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.3, stratify=y,
)

# --- Apply PCA to only the spatial (heart) features ---

num_spatial_features = heart_flat.shape[1]
num_demo_features = X_demo.shape[1]

# Split the training and test sets into spatial and demo parts
X_train_spatial = X_train[:, :num_spatial_features]
X_train_demo = X_train[:, num_spatial_features:]
X_test_spatial = X_test[:, :num_spatial_features]
X_test_demo = X_test[:, num_spatial_features:]

# Standardize spatial features before PCA
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_spatial_scaled = scaler.fit_transform(X_train_spatial)
X_test_spatial_scaled = scaler.transform(X_test_spatial)

pca = PCA(n_components=20)
X_train_spatial_pca = pca.fit_transform(X_train_spatial_scaled)
X_test_spatial_pca = pca.transform(X_test_spatial_scaled)

print("PCA spatial train shape:", X_train_spatial_pca.shape)
print("PCA spatial test shape:", X_test_spatial_pca.shape)

demo_scaler = StandardScaler()
X_train_demo_scaled = demo_scaler.fit_transform(X_train_demo)
X_test_demo_scaled = demo_scaler.transform(X_test_demo)

# Combine PCA-transformed spatial features with raw demographics
X_train = np.hstack([X_train_spatial_pca, X_train_demo_scaled])
X_test = np.hstack([X_test_spatial_pca, X_test_demo_scaled])
print("Final combined shapes:", X_train.shape, X_test.shape)


# --- Build XGBoost model ---
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist"  
)

# --- Grid Search with 5-Fold CV for Optimal XGBoost Model ---
from sklearn.model_selection import GridSearchCV

print("\n=== XGBoost Grid Search (5-Fold CV, AUC Optimization) ===")

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
    cv=5,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best AUC from Grid Search: {grid_search.best_score_:.4f}")
print("Best Parameters:", grid_search.best_params_)

# Use the best estimator from the grid search
xgb_model = grid_search.best_estimator_

# --- Train ---
xgb_model.fit(X_train, y_train)

# --- Predict ---
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

# --- Evaluate ---
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n=== XGBoost Diagnostics ===")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print("Confusion matrix:\n", cm)


# --- Feature importance diagnostics ---

print("\n=== Feature Importance (Gain) ===")
feature_importances = xgb_model.get_booster().get_score(importance_type='gain')
sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:15]
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

# Plot top 15 features by importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=15, title='Top 15 Features by Gain')
plt.tight_layout()
plt.savefig("importance.png")


print("\n=== Final Confusion Matrix (Optimal Model) ===")
cm_final = confusion_matrix(y_test, y_pred)
print(cm_final)

# --- Extended Classification Metrics ---
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, RocCurveDisplay

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Extended Classification Metrics ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'pMI']))

# --- ROC Curve Visualization ---
print("\n=== ROC Curve ===")
RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title("ROC Curve — XGBoost Model")
plt.tight_layout()
plt.savefig("roc_curve_xgboost.png")
plt.show()