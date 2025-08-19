import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Load dataset
csv_file_path = "wustl-scada-2018.csv"

# Read CSV (Handling potential errors)
try:
    df = pd.read_csv(csv_file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {csv_file_path}")
    exit()

# Check column names
print("Columns in dataset:", df.columns)

# Ensure 'Target' column exists
if 'Target' not in df.columns:
    print("Error: 'Target' column not found in dataset.")
    exit()

# Handle missing values
df.fillna(0, inplace=True)

# Feature selection (excluding 'Target' as it's the label)
X = df.drop(columns=['Target'])
y = df['Target']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)  # Probability predictions for log loss

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy  # Loss rate
logloss = log_loss(y_test, y_pred_proba)

print(f'✅ Accuracy: {accuracy:.4f}')
print(f'❌ Error Rate (Loss Rate): {error_rate:.4f}')
print(f'📉 Log Loss: {logloss:.4f}')
print('📊 Classification Report:\n', classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
print("📌 Feature Importances:\n", feature_importance)
