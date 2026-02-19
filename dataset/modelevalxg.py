import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# -------------------------------
# 1. Load data
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

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# -------------------------------
# 3. Train-test split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)

# -------------------------------
# 4. XGBoost model
# -------------------------------

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 5. Predictions
# -------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# -------------------------------
# 6. Evaluation
# -------------------------------

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(
    y_test, y_pred, target_names=le.classes_, zero_division=0
))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(
    pd.get_dummies(y_test),
    y_prob,
    multi_class="ovr"
)

print("\n🔍 ROC-AUC:", roc_auc)
