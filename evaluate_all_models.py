"""
===========================================================
 FIBROTRACKER – COMPREHENSIVE MODEL EVALUATION
 Evaluates all ML models used in the application
===========================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

BASE = os.path.dirname(os.path.abspath(__file__))
SEPARATOR = "\n" + "=" * 70

# ─────────────────────────────────────────────────────────
# 1. FIBRO RISK SCREENING MODEL (Logistic Regression)
# ─────────────────────────────────────────────────────────
def evaluate_fibro_risk():
    print(SEPARATOR)
    print("  MODEL 1: FIBRO RISK SCREENING (Logistic Regression – Elastic Net)")
    print("=" * 70)

    df = pd.read_csv(os.path.join(BASE, "dataset", "ml_training_dataset.csv"))
    print(f"  Dataset: ml_training_dataset.csv  |  Rows: {len(df)}  |  Cols: {df.shape[1]}")

    features = [
        "WPI", "SSS", "pain_regions", "symptom_persistence",
        "secondary_score_norm", "risk_factor_fraction",
        "rf_total", "modular_total_score"
    ]
    X = df[features]
    y = df["risk_category"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    model = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        C=1.0, max_iter=5000, class_weight='balanced',
        random_state=42, multi_class='multinomial'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Classes: {list(le.classes_)}")
    print(f"  Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return acc


# ─────────────────────────────────────────────────────────
# 2. FSS FATIGUE SEVERITY MODEL (Random Forest)
# ─────────────────────────────────────────────────────────
def evaluate_fss():
    print(SEPARATOR)
    print("  MODEL 2: FSS FATIGUE SEVERITY (Random Forest)")
    print("=" * 70)

    df = pd.read_csv(os.path.join(BASE, "dataset", "model", "models", "fss", "fss.csv"))
    print(f"  Dataset: fss.csv  |  Rows: {len(df)}  |  Cols: {df.shape[1]}")

    def fss_severity(total_score):
        return 0 if total_score < 36 else 1

    df["severity"] = df["Total_Score"].apply(fss_severity)
    question_cols = [f"Q{i}" for i in range(1, 10)]

    X = df[question_cols]
    y = df["severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["No Significant Fatigue", "Severe Fatigue"],
        digits=4
    ))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return acc


# ─────────────────────────────────────────────────────────
# 3. GAD-7 ANXIETY SEVERITY MODEL (Random Forest)
# ─────────────────────────────────────────────────────────
def evaluate_gad():
    print(SEPARATOR)
    print("  MODEL 3: GAD-7 ANXIETY SEVERITY (Random Forest)")
    print("=" * 70)

    df = pd.read_csv(os.path.join(BASE, "dataset", "model", "models", "gad", "gad7.csv"))
    print(f"  Dataset: gad7.csv  |  Rows: {len(df)}  |  Cols: {df.shape[1]}")

    def gad_severity(score):
        if score <= 4: return 0
        elif score <= 9: return 1
        elif score <= 14: return 2
        else: return 3

    df["severity"] = df["score"].apply(gad_severity)

    question_cols = [f"question{i}" for i in range(1, 8)]
    time_cols = [f"time{i}" for i in range(1, 8)]

    df["avg_response_time"] = df[time_cols].mean(axis=1)
    df["max_response_time"] = df[time_cols].max(axis=1)

    X = df[question_cols + ["avg_response_time", "max_response_time"]]
    y = df["severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Minimal", "Mild", "Moderate", "Moderate-Severe"],
        digits=4
    ))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return acc


# ─────────────────────────────────────────────────────────
# 4. PHQ-9 DEPRESSION SEVERITY MODEL (Random Forest)
# ─────────────────────────────────────────────────────────
def evaluate_phq():
    print(SEPARATOR)
    print("  MODEL 4: PHQ-9 DEPRESSION SEVERITY (Random Forest)")
    print("=" * 70)

    df = pd.read_csv(os.path.join(BASE, "dataset", "model", "models", "phq", "phq9.csv"))
    print(f"  Dataset: phq9.csv  |  Rows: {len(df)}  |  Cols: {df.shape[1]}")

    def phq_severity(score):
        if score <= 4: return 0
        elif score <= 9: return 1
        elif score <= 14: return 2
        elif score <= 19: return 3
        else: return 4

    df["severity"] = df["score"].apply(phq_severity)

    question_cols = [f"question{i}" for i in range(1, 10)]
    time_cols = [f"time{i}" for i in range(1, 10)]

    df["avg_response_time"] = df[time_cols].mean(axis=1)
    df["max_response_time"] = df[time_cols].max(axis=1)

    X = df[question_cols + ["avg_response_time", "max_response_time"]]
    y = df["severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"],
        digits=4
    ))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return acc


# ─────────────────────────────────────────────────────────
# 5. PSQI SLEEP QUALITY MODEL (Random Forest)
# ─────────────────────────────────────────────────────────
def evaluate_psqi():
    print(SEPARATOR)
    print("  MODEL 5: PSQI SLEEP QUALITY (Random Forest)")
    print("=" * 70)

    df = pd.read_excel(os.path.join(BASE, "dataset", "model", "models", "psqi", "psqi.xlsx"))
    print(f"  Dataset: psqi.xlsx  |  Rows: {len(df)}  |  Cols: {df.shape[1]}")

    df = df[df["PSQI_TOT"].notna()].reset_index(drop=True)
    df["PSQI_TOT"] = pd.to_numeric(df["PSQI_TOT"], errors="coerce")
    df = df[df["PSQI_TOT"].notna()].reset_index(drop=True)

    def psqi_severity(score):
        if score <= 5: return 0
        elif score <= 10: return 1
        elif score <= 15: return 2
        else: return 3

    df["PSQI_SEVERITY"] = df["PSQI_TOT"].apply(psqi_severity)

    # Remove classes with <2 samples
    counts = df["PSQI_SEVERITY"].value_counts()
    valid_classes = counts[counts >= 2].index
    df = df[df["PSQI_SEVERITY"].isin(valid_classes)].reset_index(drop=True)

    features = ["PSQIDURAT", "PSQILATEN", "PSQIHSE", "PSQIDISTB",
                 "qx_psqi_06", "qx_psqi_07", "PSQIDAYDYS"]
    X = df[features].fillna(df[features].median())
    y = df["PSQI_SEVERITY"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        random_state=42, class_weight="balanced"
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    severity_names = {0: "Good Sleep", 1: "Mild Poor", 2: "Moderate Poor", 3: "Severe Poor"}
    unique_labels = sorted(y_test.unique())
    target_names = [severity_names.get(l, f"Class {l}") for l in unique_labels]

    print(f"\n  Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return acc


# ─────────────────────────────────────────────────────────
# 6. PSS-14 STRESS SEVERITY MODEL (Random Forest)
# ─────────────────────────────────────────────────────────
def evaluate_pss():
    print(SEPARATOR)
    print("  MODEL 6: PSS-14 STRESS SEVERITY (Random Forest)")
    print("=" * 70)

    df = pd.read_csv(os.path.join(BASE, "dataset", "model", "models", "pss", "pss.csv"))
    print(f"  Dataset: pss.csv  |  Rows: {len(df)}  |  Cols: {df.shape[1]}")

    def pss_severity(score):
        if score <= 13: return 0
        elif score <= 26: return 1
        else: return 2

    df["severity"] = df["score"].apply(pss_severity)

    question_cols = [f"question{i}" for i in range(1, 15)]
    time_cols = [f"time{i}" for i in range(1, 15)]

    df["avg_response_time"] = df[time_cols].mean(axis=1)
    df["max_response_time"] = df[time_cols].max(axis=1)

    X = df[question_cols + ["avg_response_time", "max_response_time"]]
    y = df["severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Low Stress", "Moderate Stress", "High Stress"],
        digits=4
    ))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return acc


# ─────────────────────────────────────────────────────────
# MAIN: RUN ALL EVALUATIONS
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "🔬" * 35)
    print("  FIBROTRACKER – FULL MODEL EVALUATION SUITE")
    print("🔬" * 35)

    results = {}
    results["Fibro Risk (LR)"]  = evaluate_fibro_risk()
    results["FSS Fatigue (RF)"]  = evaluate_fss()
    results["GAD-7 Anxiety (RF)"] = evaluate_gad()
    results["PHQ-9 Depression (RF)"] = evaluate_phq()
    results["PSQI Sleep (RF)"]  = evaluate_psqi()
    results["PSS-14 Stress (RF)"] = evaluate_pss()

    # SUMMARY TABLE
    print(SEPARATOR)
    print("  📊  SUMMARY – ALL MODEL ACCURACIES")
    print("=" * 70)
    print(f"  {'Model':<30} {'Accuracy':>10}")
    print(f"  {'-'*30} {'-'*10}")
    for name, acc in results.items():
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {name:<30} {acc:>8.2%}  {bar}")
    print(SEPARATOR)
    avg = np.mean(list(results.values()))
    print(f"\n  Average Accuracy: {avg:.2%}")
    print(f"  Best Model     : {max(results, key=results.get)} ({max(results.values()):.2%})")
    print(f"  Worst Model    : {min(results, key=results.get)} ({min(results.values()):.2%})")
    print("\n  ✅ Evaluation complete.\n")
