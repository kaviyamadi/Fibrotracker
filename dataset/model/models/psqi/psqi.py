# =====================================================
# PSQI SEVERITY: CALCULATION + ML TRAINING + SAVE MODEL
# Dataset name: psqi.xlsx
# =====================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------- 1. LOAD DATA --------------------
DATA_FILE = "psqi.xlsx"   # Excel file
df = pd.read_excel(DATA_FILE)   # read Excel instead of CSV

print("Dataset loaded successfully")
print("Initial shape:", df.shape)

# -------------------- 2. CLEAN DATA -------------------
# Keep only valid PSQI scores
df = df[df["PSQI_TOT"].notna()].reset_index(drop=True)
df["PSQI_TOT"] = pd.to_numeric(df["PSQI_TOT"], errors="coerce")
df = df[df["PSQI_TOT"].notna()].reset_index(drop=True)

print("After cleaning:", df.shape)

# -------------------- 3. DEFINE PSQI SEVERITY ----------------
def psqi_severity(score):
    if score <= 5:
        return 0      # Good Sleep
    elif score <= 10:
        return 1      # Mild Poor Sleep
    elif score <= 15:
        return 2      # Moderate Poor Sleep
    else:
        return 3      # Severe Poor Sleep

df["PSQI_SEVERITY"] = df["PSQI_TOT"].apply(psqi_severity)

# -------------------- 3b. HANDLE VERY SMALL CLASSES ----------------
# Remove classes with <2 samples to avoid stratify error
counts = df["PSQI_SEVERITY"].value_counts()
valid_classes = counts[counts >= 2].index
df = df[df["PSQI_SEVERITY"].isin(valid_classes)].reset_index(drop=True)

# -------------------- 4. SELECT FEATURES ----------------
features = [
    "PSQIDURAT",
    "PSQILATEN",
    "PSQIHSE",
    "PSQIDISTB",
    "qx_psqi_06",
    "qx_psqi_07",
    "PSQIDAYDYS"
]

X = df[features]
y = df["PSQI_SEVERITY"]

# Handle missing feature values
X = X.fillna(X.median())

# -------------------- 5. TRAIN-TEST SPLIT ----------------
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # fallback if stratify fails
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Warning: Stratify failed, doing simple train-test split.")

# -------------------- 6. FEATURE SCALING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- 7. MODEL TRAINING ----------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# -------------------- 8. MODEL EVALUATION ----------------
y_pred = model.predict(X_test_scaled)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------- 9. SAVE MODEL & SCALER ----------------
joblib.dump(model, "psqi_severity_model.pkl")
joblib.dump(scaler, "psqi_scaler.pkl")

print("\nModel saved as: psqi_severity_model.pkl")
print("Scaler saved as: psqi_scaler.pkl")

# -------------------- 10. SAVE FINAL DATASET ----------------
df.to_excel("psqi_with_severity.xlsx", index=False)
print("Final dataset saved as: psqi_with_severity.xlsx")

# =====================================================
# END OF SCRIPT
# =====================================================
