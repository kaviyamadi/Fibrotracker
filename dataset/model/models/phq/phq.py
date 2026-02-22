# ================================
# PHQ-9 SEVERITY CLASSIFICATION
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

# Replace with your actual file path
df = pd.read_csv("phq9.csv")

# -------------------------------
# 2. SEVERITY LABEL FUNCTION
# -------------------------------

def phq_severity(score):
    if score <= 4:
        return 0  # Minimal
    elif score <= 9:
        return 1  # Mild
    elif score <= 14:
        return 2  # Moderate
    elif score <= 19:
        return 3  # Moderately Severe
    else:
        return 4  # Severe

# Apply severity labeling
df["severity"] = df["score"].apply(phq_severity)

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------

question_cols = [f"question{i}" for i in range(1, 10)]
time_cols = [f"time{i}" for i in range(1, 10)]

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
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------
# 6. MODEL EVALUATION
# -------------------------------

y_pred = model.predict(X_test)

print("\n===== PHQ-9 MODEL EVALUATION =====\n")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=[
        "Minimal",
        "Mild",
        "Moderate",
        "Moderately Severe",
        "Severe"
    ]
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 7. SAVE MODEL
# -------------------------------

joblib.dump(model, "phq_severity_model.pkl")
print("\nModel saved as: phq_severity_model.pkl")

# -------------------------------
# 8. PREDICTION FUNCTION
# -------------------------------

def predict_phq_severity(input_dict):
    """
    input_dict should contain:
    question1..question9, time1..time9
    """
    temp_df = pd.DataFrame([input_dict])

    temp_df["avg_response_time"] = temp_df[time_cols].mean(axis=1)
    temp_df["max_response_time"] = temp_df[time_cols].max(axis=1)

    features = temp_df[question_cols + ["avg_response_time", "max_response_time"]]
    pred_class = model.predict(features)[0]
    pred_prob = model.predict_proba(features).max()

    severity_map = {
        0: "Minimal",
        1: "Mild",
        2: "Moderate",
        3: "Moderately Severe",
        4: "Severe"
    }

    return {
        "Predicted_Severity": severity_map[pred_class],
        "Confidence": round(float(pred_prob), 3)
    }
import joblib

joblib.dump(model, "phq_severity_model.pkl")
