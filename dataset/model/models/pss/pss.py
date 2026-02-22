# ================================
# PSS-14 SEVERITY CLASSIFICATION
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

df = pd.read_csv("pss.csv")   # update path if needed

# -------------------------------
# 2. PSS SEVERITY LABEL FUNCTION
# -------------------------------

def pss_severity(score):
    if score <= 13:
        return 0  # Low stress
    elif score <= 26:
        return 1  # Moderate stress
    else:
        return 2  # High stress

df["severity"] = df["score"].apply(pss_severity)

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------

question_cols = [f"question{i}" for i in range(1, 15)]
time_cols = [f"time{i}" for i in range(1, 15)]

# Aggregate timing features
df["avg_response_time"] = df[time_cols].mean(axis=1)
df["max_response_time"] = df[time_cols].max(axis=1)

# Final feature set
X = df[question_cols + ["avg_response_time", "max_response_time"]]
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
    n_estimators=400,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------
# 6. MODEL EVALUATION
# -------------------------------

y_pred = model.predict(X_test)

print("\n===== PSS-14 MODEL EVALUATION =====\n")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=[
        "Low Stress",
        "Moderate Stress",
        "High Stress"
    ]
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 7. SAVE MODEL
# -------------------------------

joblib.dump(model, "pss14_severity_model.pkl")
print("\nModel saved as: pss14_severity_model.pkl")

# -------------------------------
# 8. PREDICTION FUNCTION
# -------------------------------

def predict_pss_severity(input_dict):
    """
    input_dict must contain:
    question1..question14
    time1..time14
    """

    temp_df = pd.DataFrame([input_dict])

    temp_df["avg_response_time"] = temp_df[time_cols].mean(axis=1)
    temp_df["max_response_time"] = temp_df[time_cols].max(axis=1)

    features = temp_df[question_cols + ["avg_response_time", "max_response_time"]]

    pred_class = model.predict(features)[0]
    pred_prob = model.predict_proba(features).max()

    severity_map = {
        0: "Low stress",
        1: "Moderate stress",
        2: "High stress"
    }

    return {
        "Predicted_Severity": severity_map[pred_class],
        "Confidence": round(float(pred_prob), 3)
    }
import joblib
model = joblib.load("gad7_severity_model.pkl")
