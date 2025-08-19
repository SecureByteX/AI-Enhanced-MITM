import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# Generating synthetic data (replace with real data)
np.random.seed(42)
data_size = 1000

# Dummy feature data
X = np.random.randn(data_size, 5)  # 5 features

# Dummy labels (0 = normal, 1 = suspicious)
y = np.random.randint(0, 2, size=data_size)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the model
gb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = gb_model.predict(X_test_scaled)
y_pred_proba = gb_model.predict_proba(X_test_scaled)  # Get probabilities for log loss

# Compute Accuracy and Loss Rate
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy  # Loss rate
logloss = log_loss(y_test, y_pred_proba)  # Log loss

# Print results
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"❌ Error Rate (Loss Rate): {error_rate:.4f}")
print(f"📉 Log Loss: {logloss:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
