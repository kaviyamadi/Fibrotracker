import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load dataset
# -------------------------------

df = pd.read_csv("ml_training_dataset.csv")

# -------------------------------
# 2. Feature selection
# -------------------------------

features = [
    "modular_total_score",
    "primary_scaled",
    "secondary_score_norm",
    "risk_factor_fraction",
    "WPI",
    "SSS",
    "pain_regions",
    "symptom_persistence",
    "rf_total"
]

X = df[features]
y = df["risk_category"]  # For classification

# -------------------------------
# 3. Train-test split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 4. Feature scaling
# -------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 5. Logistic Regression
# -------------------------------

model = LogisticRegression(
    penalty="l2",        # L2 regularization
    solver="liblinear",
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# -------------------------------
# 6. Predictions
# -------------------------------

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# -------------------------------
# 7. Evaluation
# -------------------------------

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 8. ROC-AUC (One-vs-Rest)
# -------------------------------

y_test_bin = pd.get_dummies(y_test)
roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")

print("\n🔍 ROC-AUC Score:", roc_auc)
