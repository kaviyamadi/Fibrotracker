import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Load dataset
# -------------------------------

df = pd.read_csv("ml_training_dataset.csv")

# -------------------------------
# 2. Feature selection (NO rule leakage)
# -------------------------------

features = [
    "WPI",
    "SSS",
    "pain_regions",
    "symptom_persistence",
    "secondary_score_norm",
    "risk_factor_fraction",
    "rf_total"
]

X = df[features]
y = df["risk_category"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------------
# 3. Train-test split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# -------------------------------
# 4. Random Forest Model
# -------------------------------

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

# -------------------------------
# 5. Predictions
# -------------------------------

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)

# -------------------------------
# 6. Evaluation
# -------------------------------

print("✅ Accuracy:", accuracy_score(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    zero_division=0
))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(
    pd.get_dummies(y_test),
    y_prob,
    multi_class="ovr"
)

print("\n🔍 ROC-AUC Score:", roc_auc)
