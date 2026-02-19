import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
print("Loading dataset...")
try:
    df = pd.read_csv("dataset/ml_training_dataset.csv")
except FileNotFoundError:
    # Try alternate path if running from root
    df = pd.read_csv("g:/Fibrotracker/dataset/ml_training_dataset.csv")

# 2. Feature selection
# Matching features used in modelevallog.py/modelevalrand.py
features = [
    "WPI",
    "SSS",
    "pain_regions",
    "symptom_persistence",
    "secondary_score_norm",
    "risk_factor_fraction",
    "rf_total"
]

print(f"Training on features: {features}")

X = df[features]
y = df["risk_category"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# 4. Random Forest Model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

# 5. Evaluation
accuracy = rf_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# 6. Save Model and Encoder
print("Saving model and label encoder...")
joblib.dump(rf_model, "fibro_risk_model.pkl")
joblib.dump(le, "fibro_risk_le.pkl")

print("✅ Model saved as 'fibro_risk_model.pkl'")
print("✅ Label Encoder saved as 'fibro_risk_le.pkl'")
