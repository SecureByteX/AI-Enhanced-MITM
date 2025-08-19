import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Example network traffic data (replace this with your actual data)
# Columns could include 'packet_size', 'protocol_type', 'src_ip', 'dest_ip', 'time_diff', 'connection_attempts', etc.
# For this example, we'll generate dummy data

# Generating synthetic data (replace with real data in practice)
np.random.seed(42)
data_size = 1000

# Dummy feature data (e.g., packet_size, time_diff between packets, etc.)
X = np.random.randn(data_size, 5)  # 5 features, replace with actual network data

# Dummy labels (1 for suspicious traffic, 0 for normal traffic)
y = np.random.randint(0, 2, size=data_size)  # 0 = normal, 1 = suspicious (MITM-like)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the data (Gradient Boosting can be sensitive to scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the model
gb_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = gb_model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optionally, you could also save the model for later use
import joblib
joblib.dump(gb_model, 'mitm_detection_model.pkl')

# Test the model with some new data (use your actual live traffic data here)
# new_data = np.random.randn(1, 5)  # Dummy new data for prediction
# new_data_scaled = scaler.transform(new_data)
# prediction = gb_model.predict(new_data_scaled)
# print("Predicted Class:", prediction)
