import numpy as np
import pandas as pd
import glob, os
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report, RocCurveDisplay
)
import xgboost as xgb
from sklearn.svm import SVC
from scipy.stats import uniform, randint


# === Load and flatten all frames ===
shape_dir = "/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/training_data/mixed_samples"
files = sorted(glob.glob(os.path.join(shape_dir, "*.npy")))

# === Remove known corrupted samples ===
corrupted_files = [
    '426pMI.npy', '445pMI.npy', '846healthy.npy', '289healthy.npy', '208healthy.npy', '595healthy.npy',
    '410pMI.npy', '390healthy.npy', '814pMI.npy', '693healthy.npy', '754pMI.npy', '841pMI.npy',
    '8healthy.npy', '174pMI.npy', '853pMI.npy', '175healthy.npy', '875pMI.npy', '308healthy.npy',
    '734healthy.npy', '793healthy.npy', '424pMI.npy', '573healthy.npy', '258healthy.npy', '21pMI.npy',
    '163healthy.npy', '172pMI.npy', '776pMI.npy', '335healthy.npy', '139pMI.npy', '794healthy.npy',
    '508pMI.npy', '502healthy.npy', '699pMI.npy', '292healthy.npy', '479healthy.npy', '68pMI.npy',
    '280pMI.npy', '774pMI.npy', '621healthy.npy', '385pMI.npy', '70healthy.npy', '485healthy.npy',
    '448pMI.npy', '378pMI.npy', '217pMI.npy', '591pMI.npy', '738pMI.npy', '638pMI.npy', '345healthy.npy',
    '673pMI.npy', '86pMI.npy', '313pMI.npy', '503healthy.npy', '899healthy.npy', '80healthy.npy',
    '539healthy.npy', '138pMI.npy', '91healthy.npy', '75pMI.npy', '119healthy.npy', '20pMI.npy',
    '872pMI.npy', '181healthy.npy', '597pMI.npy', '203pMI.npy', '658healthy.npy', '439healthy.npy', '564pMI.npy'
]

# Filter out corrupted files
files = [f for f in files if os.path.basename(f) not in corrupted_files]
print(f"Removed {len(corrupted_files)} corrupted samples. Remaining valid samples: {len(files)}")

heart_samples = [np.load(f) for f in files]  # each shape (10, 18000, 4)
heart_samples = np.array(heart_samples)
print("Heart samples shape:", heart_samples.shape)

# Extract xyz and flatten
heart_xyz = heart_samples[..., :3]
heart_flat = heart_xyz.reshape(900, -1)
print("Flattened heart shape:", heart_flat.shape)

# === Load demographics ===
demo_df = pd.read_csv("/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/training_data/mixed_demographics.csv")
demo_df["height"] = demo_df["height"] / 100  # convert cm to m

# Include 'sex' column as numeric (convert True/False to 1/0, where True = Male)
demo_df['sex'] = demo_df['sex'].apply(lambda x: 1 if x is True else 0)
demo_features = ['age', 'BMI', 'height', 'weight', 'diastolic_BP', 'systolic_BP', 'sex']
X_demo = demo_df[demo_features].values
y = demo_df['MI'].map({'pMI': 1, 'healthy': 0}).values
print("Included 'sex' as a demographic feature (1=Male, 0=Female)")

# === PCA for heart data only ===
num_spatial_features = heart_flat.shape[1]
num_demo_features = X_demo.shape[1]

scaler = StandardScaler()
heart_scaled = scaler.fit_transform(heart_flat)

pca = PCA(n_components=50)
heart_pca = pca.fit_transform(heart_scaled)

# Scale demographics
demo_scaler = StandardScaler()
X_demo_scaled = demo_scaler.fit_transform(X_demo)

# Combine final features
X_final = np.hstack([heart_pca, X_demo_scaled])
print("Final training data shape:", X_final.shape)

# === Define base models ===
log_reg = LogisticRegression(max_iter=50000, solver='lbfgs', tol=1e-3)
rf = RandomForestClassifier()
xgb_model = xgb.XGBClassifier()


print("\n=== Tuning Base Models with Randomized Search and Cross-Validation ===")


# Random Forest random search
rf_params = {'n_estimators': randint(100, 600), 'max_depth': randint(3, 12), 'min_samples_split': randint(2, 6)}
rf_search = RandomizedSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_iter=200, n_jobs=-1, random_state=42)
rf_search.fit(X_final, y)
rf = rf_search.best_estimator_
print("Best Random Forest params:", rf_search.best_params_)

# XGBoost random search
xgb_params = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 12),
    'learning_rate': uniform(0.01, 0.1),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}
xgb_search = RandomizedSearchCV(xgb_model, xgb_params, cv=5, scoring='roc_auc', n_iter=200, n_jobs=-1, verbose=2, random_state=42)
xgb_search.fit(X_final, y)
xgb_model = xgb_search.best_estimator_
print("Best XGBoost params:", xgb_search.best_params_)

print("\nUsing optimized models for ensemble...")

# === Stacking Ensemble ===
stacked_model = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb_model),
    ],
    final_estimator=LogisticRegression(max_iter=5000, solver='lbfgs'),
    cv=5,
    n_jobs=-1
)

print("\n=== Training Stacking Ensemble (RF + XGB, Meta LR) on Full Dataset ===")
stacked_model.fit(X_final, y)

# === Load and predict on new unseen data ===
new_data_dir = "/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/test_samples/test_samples"
new_files = sorted(glob.glob(os.path.join(new_data_dir, "*.npy")))

new_samples = [np.load(f) for f in new_files]
new_samples = np.array(new_samples)
print("New test samples shape:", new_samples.shape)

# === Prepare new spatial data for model input (no outlier filtering) ===
heart_col = new_samples[..., :3]
new_flat = heart_col.reshape(len(new_samples), -1)
print("New flattened data shape for model input:", new_flat.shape)

# Load new demographics
new_demo_df = pd.read_csv("/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/test_samples/test.csv")
new_demo_df["height"] = new_demo_df["height"] / 100
new_demo_df["sex"] = new_demo_df["sex"].astype(int)
new_demo_features = ['age', 'BMI', 'height', 'weight', 'diastolic_BP', 'systolic_BP', 'sex']
X_new_demo = new_demo_df[new_demo_features].values


# Truncate to match PCA input feature size from training
expected_pca_features = 499200
if new_flat.shape[1] > expected_pca_features:
    print(f"Truncating new_flat from {new_flat.shape[1]} to {expected_pca_features} features to match PCA input size.")
    new_flat = new_flat[:, :expected_pca_features]
elif new_flat.shape[1] < expected_pca_features:
    raise ValueError(f"new_flat has fewer features ({new_flat.shape[1]}) than PCA expects ({expected_pca_features}). Check preprocessing steps.")

scaler_new = StandardScaler()
new_spatial_scaled = scaler_new.fit_transform(new_flat)
new_spatial_pca = pca.transform(new_spatial_scaled)
scaler_new2 = StandardScaler()
new_demo_scaled = scaler_new2.fit_transform(X_new_demo)
X_new_final = np.hstack([new_spatial_pca, new_demo_scaled])

# Predict with stacked model
y_new_prob = stacked_model.predict_proba(X_new_final)[:, 1]
y_new_pred = (y_new_prob > 0.5).astype(int)

# Save predictions
new_demo_df["Predicted_MI"] = y_new_pred
new_demo_df["Predicted_Prob"] = y_new_prob
new_demo_df.to_csv("ensemble_predictions_new_data.csv", index=False)

# Extract only the Predicted_MI column
predicted_mi = new_demo_df[["Predicted_MI"]]

# Save to a new CSV file
predicted_mi.to_csv("predicted_MI_only.csv", index=False)

print("\nPredictions saved to ensemble_predictions_new_data.csv")