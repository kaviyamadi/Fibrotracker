# ================================
# GAD-7 SEVERITY CLASSIFICATION
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

df = pd.read_csv("gad7.csv")   # update path if needed

# -------------------------------
# 2. GAD-7 SEVERITY LABEL FUNCTION
# -------------------------------

def gad_severity(score):
    if score <= 4:
        return 0  # Minimal
    elif score <= 9:
        return 1  # Mild
    elif score <= 14:
        return 2  # Moderate
    else:
        return 3  # Moderate to Severe

df["severity"] = df["score"].apply(gad_severity)

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------

question_cols = [f"question{i}" for i in range(1, 8)]
time_cols = [f"time{i}" for i in range(1, 8)]

# Aggregate response timing
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

print("\n===== GAD-7 MODEL EVALUATION =====\n")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=[
        "Minimal",
        "Mild",
        "Moderate",
        "Moderate to Severe"
    ]
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 7. SAVE MODEL
# -------------------------------

joblib.dump(model, "gad7_severity_model.pkl")
print("\nModel saved as: gad7_severity_model.pkl")

# -------------------------------
# 8. PREDICTION FUNCTION
# -------------------------------

def predict_gad_severity(input_dict):
    """
    input_dict must contain:
    question1..question7
    time1..time7
    """

    temp_df = pd.DataFrame([input_dict])

    temp_df["avg_response_time"] = temp_df[time_cols].mean(axis=1)
    temp_df["max_response_time"] = temp_df[time_cols].max(axis=1)

    features = temp_df[question_cols + ["avg_response_time", "max_response_time"]]

    pred_class = model.predict(features)[0]
    pred_prob = model.predict_proba(features).max()

    severity_map = {
        0: "Minimal anxiety",
        1: "Mild anxiety",
        2: "Moderate anxiety",
        3: "Moderate to severe anxiety"
    }

    return {
        "Predicted_Severity": severity_map[pred_class],
        "Confidence": round(float(pred_prob), 3)
    }
