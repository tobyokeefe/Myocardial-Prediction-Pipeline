import numpy as np
import pandas as pd
import glob, os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report, RocCurveDisplay
)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# === Load and flatten all frames ===
shape_dir = "/Users/tobyokeefe/git/Myocardial-Prediction-Pipeline/data/training_data/mixed_samples"
files = sorted(glob.glob(os.path.join(shape_dir, "*.npy")))

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
demo_df['sex'] = demo_df['sex'].astype(int)
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
print("Scaled demographic data independently of PCA features.")

# === Define base models ===
log_reg = LogisticRegression(max_iter=50000, solver='lbfgs', tol=1e-3)
rf = RandomForestClassifier()
xgb_model = xgb.XGBClassifier()


print("\n=== Tuning Base Models with Grid Search and Cross-Validation ===")

# Logistic Regression grid
log_reg_params = {'C': [0.01, 0.1, 0.5, 1.0, 10.0], 'penalty': ['l2'], 'solver': ['lbfgs']}
log_search = GridSearchCV(log_reg, log_reg_params, cv=5, scoring='roc_auc', n_jobs=-1)
log_search.fit(X_train_final, y_train)
log_reg = log_search.best_estimator_
print("Best Logistic Regression params:", log_search.best_params_)

# Random Forest grid
rf_params = {'n_estimators': [200, 300, 400, 600], 'max_depth': [4, 6, 8, 10], 'min_samples_split': [2, 4, 6]}
rf_search = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
rf_search.fit(X_train_final, y_train)
rf = rf_search.best_estimator_
print("Best Random Forest params:", rf_search.best_params_)

# XGBoost grid
xgb_params = {
    'n_estimators': [10, 20, 50, 100, 200],
    'max_depth': [4, 6, 8, 10, 12],
    'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_search = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
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

# === Ensemble (soft voting) ===
ensemble = VotingClassifier(
    estimators=[('lr', log_reg), ('rf', rf), ('xgb', xgb_model)],
    voting='soft',  # uses predicted probabilities
    weights=[1, 2, 2]  # weight tree models slightly higher
)

print("\n=== Training Ensemble (LogReg + RF + XGB) ===")
ensemble.fit(X_train_final, y_train)

# --- Predictions ---
y_pred_prob = ensemble.predict_proba(X_test_final)[:, 1]
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

# --- ROC Curve ---
RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title("ROC Curve — Ensemble (LogReg + RF + XGB)")
plt.tight_layout()
plt.savefig("ensemble_roc.png")
plt.show()

# --- Feature importance (XGBoost only) ---
xgb_fitted = ensemble.named_estimators_['xgb']  # retrieve trained XGB model
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_fitted, importance_type='gain', max_num_features=15)
plt.title("Top 15 Feature Importances (XGBoost component)")
plt.tight_layout()
plt.savefig("ensemble_xgb_importance.png")
plt.show()