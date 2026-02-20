import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. Load dataset
print("Loading dataset...")
try:
    df = pd.read_csv("dataset/ml_training_dataset.csv")
except FileNotFoundError:
    df = pd.read_csv("g:/Fibrotracker/dataset/ml_training_dataset.csv")

# 2. Feature selection (includes modular_total_score per spec)
features = [
    "WPI",
    "SSS",
    "pain_regions",
    "symptom_persistence",
    "secondary_score_norm",
    "risk_factor_fraction",
    "rf_total",
    "modular_total_score"
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

# 4. Logistic Regression with L1/L2 (Elastic Net) regularization per spec
print("Training Logistic Regression model (elastic net)...")
lr_model = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,      # Balance between L1 and L2
    C=1.0,
    max_iter=5000,
    class_weight='balanced',
    random_state=42,
    multi_class='multinomial'
)

lr_model.fit(X_train, y_train)

# 5. Evaluation
accuracy = lr_model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
y_pred = lr_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 6. Save Model and Encoder
print("Saving model and label encoder...")
joblib.dump(lr_model, "fibro_risk_model.pkl")
joblib.dump(le, "fibro_risk_le.pkl")

print("✅ Model saved as 'fibro_risk_model.pkl'")
print("✅ Label Encoder saved as 'fibro_risk_le.pkl'")
