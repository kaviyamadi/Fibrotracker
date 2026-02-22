# ================================
# FSS SEVERITY CLASSIFICATION
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

# -------------------------------
# 1. LOAD DATA
# -------------------------------

df = pd.read_csv("fss.csv")   # columns: Q1..Q9, Total_Score

# -------------------------------
# 2. FSS SEVERITY LABEL FUNCTION
# -------------------------------

def fss_severity(total_score):
    if total_score < 36:
        return 0  # No significant fatigue
    else:
        return 1  # Severe fatigue

df["severity"] = df["Total_Score"].apply(fss_severity)

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------

question_cols = [f"Q{i}" for i in range(1, 10)]

X = df[question_cols]
y = df["severity"]

# -------------------------------
# 4. TRAIN-TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 5. MODEL TRAINING
# -------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------
# 6. MODEL EVALUATION
# -------------------------------

y_pred = model.predict(X_test)

print("\n===== FSS MODEL EVALUATION =====\n")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=[
        "No Significant Fatigue",
        "Severe Fatigue"
    ]
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 7. SAVE MODEL
# -------------------------------

joblib.dump(model, "fss_severity_model.pkl")
print("\nModel saved as: fss_severity_model.pkl")

# -------------------------------
# 8. PREDICTION FUNCTION
# -------------------------------

def predict_fss_severity(input_dict):
    """
    input_dict must contain:
    Q1..Q9
    """

    temp_df = pd.DataFrame([input_dict])
    features = temp_df[question_cols]

    pred_class = model.predict(features)[0]
    pred_prob = model.predict_proba(features).max()

    severity_map = {
        0: "No significant fatigue",
        1: "Clinically significant fatigue"
    }

    return {
        "Predicted_Severity": severity_map[pred_class],
        "Confidence": round(float(pred_prob), 3)
    }
