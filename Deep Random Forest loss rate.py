import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Define dataset path
data_path = "wustl-scada-2018.csv"

print("🔍 Checking file existence...")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")
print("✅ File found!")

# Load dataset efficiently
print("📥 Loading dataset...")
df = pd.read_csv(data_path, low_memory=False)
print("✅ Dataset loaded successfully!")

# Ensure 'Target' column exists
if 'Target' not in df.columns:
    raise ValueError("Dataset does not contain 'Target' column")

# Handling missing values
if df.isnull().sum().sum() > 0:
    print("⚠️ Missing values detected. Filling with column means...")
    df.fillna(df.mean(), inplace=True)
    print("✅ Missing values handled.")

# Convert categorical features
if df.select_dtypes(include=['object']).shape[1] > 0:
    print("🔄 Converting categorical features to numerical...")
    df = pd.get_dummies(df)
    print("✅ Categorical conversion complete.")

# Feature selection
X = df.drop(columns=['Target'])
y = df['Target']  

# Splitting data (reduce size for speed during testing)
print("✂️ Splitting dataset into training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure features are aligned
X_train, X_test = X_train.align(X_test, join='inner', axis=1)
print("✅ Feature alignment complete.")

# Define models
rf_model = RandomForestClassifier(n_estimators=100, warm_start=True, n_jobs=-1, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
print("🌲 Using RandomForestClassifier and XGBoostClassifier.")

# Train models
print("🚀 Training models...")
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
print("✅ Training complete.")

# Predictions
print("🤖 Making predictions...")
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Probabilistic Predictions (For Log Loss Calculation)
y_pred_proba_rf = rf_model.predict_proba(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)

# Evaluate RandomForest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
error_rate_rf = 1 - accuracy_rf
logloss_rf = log_loss(y_test, y_pred_proba_rf)

print(f'🎯 RandomForest Accuracy: {accuracy_rf:.4f}')
print(f'❌ RandomForest Error Rate: {error_rate_rf:.4f}')
print(f'📉 RandomForest Log Loss: {logloss_rf:.4f}')
print('📊 RandomForest Classification Report:')
print(classification_report(y_test, y_pred_rf))

# Evaluate XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
error_rate_xgb = 1 - accuracy_xgb
logloss_xgb = log_loss(y_test, y_pred_proba_xgb)

print(f'🎯 XGBoost Accuracy: {accuracy_xgb:.4f}')
print(f'❌ XGBoost Error Rate: {error_rate_xgb:.4f}')
print(f'📉 XGBoost Log Loss: {logloss_xgb:.4f}')
print('📊 XGBoost Classification Report:')
print(classification_report(y_test, y_pred_xgb))

# Feature importance (RandomForest)
feature_importance_rf = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("🌟 RandomForest Feature Importances:")
print(feature_importance_rf.head(10))

# Feature importance (XGBoost)
feature_importance_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("🌟 XGBoost Feature Importances:")
print(feature_importance_xgb.head(10))
