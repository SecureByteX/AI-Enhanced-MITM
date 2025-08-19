import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

# Load the dataset
def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            print("❌ Error: File not found!")
            exit(1)
            
        print(f"📂 Loading dataset from: {file_path}")  
        data = pd.read_csv(file_path)

        print(f"✅ Dataset Loaded Successfully! Shape: {data.shape}")  

        # Check for missing values
        if data.isnull().sum().sum() > 0:
            print("⚠️ Warning: Dataset contains missing values. Filling with mean...")
            data.fillna(data.mean(), inplace=True)

        # Convert categorical variables
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        if X.select_dtypes(include=['object']).shape[1] > 0:
            print("⚠️ Warning: Dataset contains categorical values. Converting to numeric...")
            X = pd.get_dummies(X)

        return X, y
    except Exception as e:
        print("❌ Error loading data:", e)
        exit(1)

# Preprocess features
def preprocess_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Hyperparameter tuning (Optional: Can be skipped to save time)
def tune_model(ert_model, param_grid, X_train, y_train):
    print("⏳ Tuning Hyperparameters (This may take time)...")
    grid_search = GridSearchCV(ert_model, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"✅ Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Evaluate the model
def evaluate_model(y_test, y_pred, y_pred_proba):
    print("\n📊 Model Evaluation:")
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy  # Loss Rate
    logloss = log_loss(y_test, y_pred_proba)

    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"❌ Loss Rate: {error_rate:.4f}")
    print(f"📉 Log Loss: {logloss:.4f}")
    print("📄 Classification Report:\n", classification_report(y_test, y_pred))
    print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Main script execution
if __name__ == "__main__":
    print("🚀 Script Started!")

    # Define dataset path
    file_path = "wustl-scada-2018.csv"
    # Load and split the dataset
    X, y = load_data(file_path)

    print("✂️ Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔄 Ensuring training & test feature alignment...")
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)

    print("⚙️ Preprocessing Features...")
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)

    # Initialize the ExtraTreesClassifier model
    ert_model = ExtraTreesClassifier(random_state=42)

    # Hyperparameter tuning (can be skipped for speed)
    param_grid = {
        'n_estimators': [50],  
        'max_depth': [None],  
        'min_samples_split': [2]
    }
    
    # Skip tuning if you want faster results
    best_ert_model = tune_model(ert_model, param_grid, X_train_scaled, y_train)

    print("🤖 Training Model...")
    best_ert_model.fit(X_train_scaled, y_train)

    print("🔍 Making Predictions...")
    y_pred = best_ert_model.predict(X_test_scaled)
    y_pred_proba = best_ert_model.predict_proba(X_test_scaled)  # Probability predictions for log loss

    # Evaluate the model
    evaluate_model(y_test, y_pred, y_pred_proba)

    print("✅ Script Execution Completed!")
