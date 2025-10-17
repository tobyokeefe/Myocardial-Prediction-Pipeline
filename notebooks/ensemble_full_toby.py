import numpy as np
import pandas as pd
import glob, os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report, RocCurveDisplay
)
import xgboost as xgb
from sklearn.svm import SVC
from scipy.stats import uniform, randint
from sklearn.ensemble import StackingClassifier


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

# Include 'sex' column as numeric (convert True/False to 1/0)
demo_df['sex'] = demo_df['sex'].apply(lambda x: 1 if x is True else 0)
demo_features = ['age', 'BMI', 'height', 'weight', 'diastolic_BP', 'systolic_BP', 'sex']
X_demo = demo_df[demo_features].values
y = demo_df['MI'].map({'pMI': 1, 'healthy': 0}).values
print("Included 'sex' as a demographic feature (0=Male, 1=Female)")

# Combine
X_combined = np.hstack([heart_flat, X_demo])
print("Combined shape:", X_combined.shape)

# === Split into train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.1, stratify=y, random_state=42
)

# === PCA for heart data only ===
num_spatial_features = heart_flat.shape[1]
num_demo_features = X_demo.shape[1]

X_train_spatial = X_train[:, :num_spatial_features]
X_train_demo = X_train[:, num_spatial_features:]
X_test_spatial = X_test[:, :num_spatial_features]
X_test_demo = X_test[:, num_spatial_features:]

scaler = StandardScaler()
X_train_spatial_scaled = scaler.fit_transform(X_train_spatial)
X_test_spatial_scaled = scaler.transform(X_test_spatial)

pca = PCA(n_components=50)
X_train_spatial_pca = pca.fit_transform(X_train_spatial_scaled)
X_test_spatial_pca = pca.transform(X_test_spatial_scaled)

# --- Scale demographic data independently ---
demo_scaler = StandardScaler()
X_train_demo_scaled = demo_scaler.fit_transform(X_train_demo)
X_test_demo_scaled = demo_scaler.transform(X_test_demo)

# Combine PCA (already standardized) with scaled demographics
X_train_final = np.hstack([X_train_spatial_pca, X_train_demo_scaled])
X_test_final = np.hstack([X_test_spatial_pca, X_test_demo_scaled])
print("Final feature shapes:", X_train_final.shape, X_test_final.shape)
print("Scaled demographic and temporal motion data independently of PCA features.")

# === Define base models ===
log_reg = LogisticRegression(max_iter=50000, solver='lbfgs', tol=1e-3)
rf = RandomForestClassifier()
xgb_model = xgb.XGBClassifier()


print("\n=== Tuning Base Models with Randomized Search and Cross-Validation ===")

# Logistic Regression random search
log_reg_params = {
    'C': uniform(0.01, 10.0)
}
log_search = RandomizedSearchCV(log_reg, log_reg_params, cv=5, scoring='roc_auc', n_iter=1000, n_jobs=-1, random_state=42)
log_search.fit(X_train_final, y_train)
log_reg = log_search.best_estimator_
print("Best Logistic Regression params:", log_search.best_params_)

# Random Forest random search
rf_params = {
    'n_estimators': randint(100, 600),
    'max_depth': randint(3, 12),
    'min_samples_split': randint(2, 6)
}
rf_search = RandomizedSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_iter=500, n_jobs=-1, random_state=42)
rf_search.fit(X_train_final, y_train)
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
xgb_search = RandomizedSearchCV(xgb_model, xgb_params, cv=5, scoring='roc_auc', n_iter=500, n_jobs=-1, verbose=0, random_state=42)
xgb_search.fit(X_train_final, y_train)
xgb_model = xgb_search.best_estimator_
print("Best XGBoost params:", xgb_search.best_params_)

# --- Evaluate individual models before combining ---
from sklearn.metrics import classification_report
print("\n=== Evaluating Individual Models ===")

# Logistic Regression
y_pred_prob_lr = log_reg.predict_proba(X_test_final)[:, 1]
y_pred_lr = (y_pred_prob_lr > 0.5).astype(int)
print("\n--- Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_prob_lr):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr, target_names=['Healthy', 'pMI']))

# Random Forest
y_pred_prob_rf = rf.predict_proba(X_test_final)[:, 1]
y_pred_rf = (y_pred_prob_rf > 0.5).astype(int)
print("\n--- Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_prob_rf):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=['Healthy', 'pMI']))

# XGBoost
y_pred_prob_xgb = xgb_model.predict_proba(X_test_final)[:, 1]
y_pred_xgb = (y_pred_prob_xgb > 0.5).astype(int)
print("\n--- XGBoost ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_prob_xgb):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb, target_names=['Healthy', 'pMI']))

print("\nUsing optimized models for ensemble...")

# === Stacking Ensemble ===
stacked_model = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=2000, solver='lbfgs'),
    cv=10,
    n_jobs=-1
)

print("\n=== Training Stacking Ensemble (LR + RF + XGB) ===")
stacked_model.fit(X_train_final, y_train)

# --- Predictions ---
y_pred_prob = stacked_model.predict_proba(X_test_final)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

# --- Evaluation ---
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Ensemble Diagnostics ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("Confusion matrix:\n", cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'pMI']))

# === Global Feature Importance (Permutation-based for Stacking Ensemble) ===
from sklearn.inspection import permutation_importance

print("\n=== Computing Global Feature Importance (Permutation Importance) ===")

# Combine PCA and demographic feature names
pca_feature_names = [f"PCA_{i+1}" for i in range(50)]
feature_names = pca_feature_names + demo_features

# Compute permutation importances
r = permutation_importance(
    stacked_model, X_test_final, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)

# Sort and plot top 20 features
sorted_idx = r.importances_mean.argsort()[::-1][:20]
plt.figure(figsize=(10, 6))
plt.barh(np.array(feature_names)[sorted_idx], r.importances_mean[sorted_idx], xerr=r.importances_std[sorted_idx])
plt.gca().invert_yaxis()
plt.xlabel("Mean Importance (Decrease in AUC)")
plt.title("Global Feature Importance — Stacking Ensemble (Permutation Importance)")
plt.tight_layout()
plt.savefig("stacking_global_feature_importance.png")
plt.close()

print("Global feature importance plot saved as 'stacking_global_feature_importance.png'")

# --- ROC Curve ---
RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title("ROC Curve — Stacking Ensemble (RF + XGB)")
plt.tight_layout()
plt.savefig("stacking_ensemble_roc.png")
plt.show()

# --- Feature importance (XGBoost only) ---
xgb_fitted = stacked_model.named_estimators_['xgb']
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_fitted, importance_type='gain', max_num_features=15)
plt.title("Top 15 Feature Importances (XGBoost component in Stacked Model)")
plt.tight_layout()
plt.savefig("stacking_xgb_importance.png")
plt.show()