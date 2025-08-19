import pandas as pd
import numpy as np
import os
import warnings
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Track execution time
start_time = time.time()

# Define dataset path
data_path = os.path.abspath("wustl-scada-2018.csv")  # Ensures correct file path

print("🔍 Checking file existence...")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")
print("✅ File found!")

# Load dataset efficiently
print("📥 Loading dataset...")
df = pd.read_csv(data_path, low_memory=False)
print("✅ Dataset loaded successfully!")

# Ensure 'Target' column exists
if 'Target' not in df.columns:
    raise ValueError("❌ Dataset does not contain 'Target' column")

# Handling missing values properly
if df.isnull().sum().sum() > 0:
    print("⚠️ Missing values detected. Handling missing values...")
    
    # Fill numeric columns with mean
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    
    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    print("✅ Missing values handled.")

# Feature selection
X = df.drop(columns=['Target'])
y = df['Target']

# Compute class weights to handle imbalances
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

# Splitting data into training and testing sets
print("✂️ Splitting dataset into training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ensure features are aligned and fill missing values
X_train, X_test = X_train.align(X_test, join='inner', axis=1, fill_value=0)
print("✅ Feature alignment complete.")

# Define RandomForest Model
rf_classifier = RandomForestClassifier(n_jobs=-1, class_weight=class_weight_dict, random_state=42)

# Hyperparameter tuning (using RandomizedSearchCV for speed)
param_grid = {
    'n_estimators': [100, 200, 300],  # Added 300 for more options
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("🔍 Performing hyperparameter tuning...")
search = RandomizedSearchCV(
    rf_classifier, param_grid, cv=5, scoring='accuracy', n_iter=10, n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)
best_rf = search.best_estimator_
print(f"✅ Best Parameters Found: {search.best_params_}")

# Predictions
print("🤖 Making predictions...")
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)  # Probability predictions for log loss

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy  # Loss rate
logloss = log_loss(y_test, y_pred_proba)

print(f'🎯 Accuracy: {accuracy:.4f}')
print(f'❌ Error Rate (Loss Rate): {error_rate:.4f}')
print(f'📉 Log Loss: {logloss:.4f}')
print('📊 Classification Report:')
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("🌟 Top 10 Feature Importances:")
print(feature_importance.head(10))

# Execution time tracking
end_time = time.time()
print(f"🚀 Execution Time: {end_time - start_time:.2f} seconds")
