import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier

# -------------------------------
# 1. LOAD DATA
# -------------------------------
print("Loading dataset...")
data = pd.read_csv("final_dataset.csv")

print("\nInitial Shape:", data.shape)

# -------------------------------
# 2. RENAME COLUMNS (TShark → ML)
# -------------------------------
# Expected order from tshark export:
# frame.len, tcp.srcport, tcp.dstport, ip.proto,
# tcp.len, tcp.window_size, frame.time_delta, label

expected_cols = [
    "packet_size",
    "src_port",
    "dst_port",
    "protocol",
    "tcp_len",
    "tcp_window",
    "time_delta",
    "label"
]

if len(data.columns) == 8:
    data.columns = expected_cols
else:
    print("\n⚠️ Unexpected column count:", len(data.columns))
    print("Columns found:", list(data.columns))
    raise SystemExit("Fix your CSV columns before training.")

print("\nColumns:", list(data.columns))

# -------------------------------
# 3. CLEAN DATA
# -------------------------------
print("\nCleaning data...")
data = data.dropna()
data = data.drop_duplicates()

# Remove extreme values
data = data[data['packet_size'] < 10000]

print("Cleaned Shape:", data.shape)

# -------------------------------
# 4. CHECK LABEL DISTRIBUTION
# -------------------------------
print("\nLabel Distribution:")
print(data['label'].value_counts())

# -------------------------------
# 5. FEATURES & LABEL
# -------------------------------
features = [
    'packet_size',
    'src_port',
    'dst_port',
    'protocol',
    'tcp_len',
    'tcp_window',
    'time_delta'
]

X = data[features]
y = data['label']

# -------------------------------
# 6. SCALING
# -------------------------------
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 7. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------------
# 8. MODEL (XGBOOST)
# -------------------------------
print("\nTraining model...")

# Handle imbalance safely
pos = (y == 1).sum()
neg = (y == 0).sum()
scale_weight = neg / pos if pos > 0 else 1

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------
# 9. PREDICTION
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------
# 10. EVALUATION
# -------------------------------
print("\n=== MODEL EVALUATION ===")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -------------------------------
# 11. CROSS-VALIDATION
# -------------------------------
print("\nCross-validation...")

cv_scores = cross_val_score(model, X_scaled, y, cv=5, n_jobs=-1)
print("Cross-Validation Accuracy:", cv_scores.mean())

# -------------------------------
# 12. SAVE MODEL
# -------------------------------
print("\nSaving model...")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel and scaler saved successfully!")
