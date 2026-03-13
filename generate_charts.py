"""
===========================================================
 FIBROTRACKER – MODEL EVALUATION CHART GENERATOR
 Generates diverse charts for documentation purposes
===========================================================
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)

# ── Paths ────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# ── Theme ────────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
TEXT_CLR   = "#e6edf3"
GRID_CLR   = "#21262d"
ACCENT     = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff",
              "#f0883e", "#79c0ff", "#56d364", "#ff7b72"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   GRID_CLR,
    "axes.labelcolor":  TEXT_CLR,
    "text.color":       TEXT_CLR,
    "xtick.color":      TEXT_CLR,
    "ytick.color":      TEXT_CLR,
    "grid.color":       GRID_CLR,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

# ══════════════════════════════════════════════════════════
# DATA LOADING & MODEL TRAINING HELPERS
# ══════════════════════════════════════════════════════════

def _load_and_train_fibro():
    df = pd.read_csv(os.path.join(BASE, "dataset", "ml_training_dataset.csv"))
    features = ["WPI","SSS","pain_regions","symptom_persistence",
                "secondary_score_norm","risk_factor_fraction",
                "rf_total","modular_total_score"]
    X = df[features]; y = df["risk_category"]
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2,
                                           stratify=y_enc, random_state=42)
    m = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                           C=1.0, max_iter=5000, class_weight='balanced',
                           random_state=42, multi_class='multinomial')
    m.fit(Xtr, ytr)
    return m, Xtr, Xte, ytr, yte, list(le.classes_), features, X, y_enc

def _load_and_train_fss():
    df = pd.read_csv(os.path.join(BASE, "dataset","model","models","fss","fss.csv"))
    df["severity"] = df["Total_Score"].apply(lambda s: 0 if s < 36 else 1)
    cols = [f"Q{i}" for i in range(1,10)]
    X = df[cols]; y = df["severity"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = RandomForestClassifier(n_estimators=200, random_state=42,
                               class_weight="balanced", n_jobs=-1)
    m.fit(Xtr, ytr)
    names = ["No Significant Fatigue","Severe Fatigue"]
    return m, Xtr, Xte, ytr, yte, names, cols, X, y

def _load_and_train_gad():
    df = pd.read_csv(os.path.join(BASE, "dataset","model","models","gad","gad7.csv"))
    def sev(s):
        if s<=4: return 0
        elif s<=9: return 1
        elif s<=14: return 2
        else: return 3
    df["severity"] = df["score"].apply(sev)
    qc = [f"question{i}" for i in range(1,8)]
    tc = [f"time{i}" for i in range(1,8)]
    df["avg_rt"] = df[tc].mean(axis=1); df["max_rt"] = df[tc].max(axis=1)
    feats = qc + ["avg_rt","max_rt"]
    X = df[feats]; y = df["severity"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = RandomForestClassifier(n_estimators=300, random_state=42,
                               class_weight="balanced", n_jobs=-1)
    m.fit(Xtr, ytr)
    names = ["Minimal","Mild","Moderate","Moderate-Severe"]
    return m, Xtr, Xte, ytr, yte, names, feats, X, y

def _load_and_train_phq():
    df = pd.read_csv(os.path.join(BASE, "dataset","model","models","phq","phq9.csv"))
    def sev(s):
        if s<=4: return 0
        elif s<=9: return 1
        elif s<=14: return 2
        elif s<=19: return 3
        else: return 4
    df["severity"] = df["score"].apply(sev)
    qc = [f"question{i}" for i in range(1,10)]
    tc = [f"time{i}" for i in range(1,10)]
    df["avg_rt"] = df[tc].mean(axis=1); df["max_rt"] = df[tc].max(axis=1)
    feats = qc + ["avg_rt","max_rt"]
    X = df[feats]; y = df["severity"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = RandomForestClassifier(n_estimators=300, random_state=42,
                               class_weight="balanced", n_jobs=-1)
    m.fit(Xtr, ytr)
    names = ["Minimal","Mild","Moderate","Moderately Severe","Severe"]
    return m, Xtr, Xte, ytr, yte, names, feats, X, y

def _load_and_train_psqi():
    df = pd.read_excel(os.path.join(BASE, "dataset","model","models","psqi","psqi.xlsx"))
    df = df[df["PSQI_TOT"].notna()].reset_index(drop=True)
    df["PSQI_TOT"] = pd.to_numeric(df["PSQI_TOT"], errors="coerce")
    df = df[df["PSQI_TOT"].notna()].reset_index(drop=True)
    def sev(s):
        if s<=5: return 0
        elif s<=10: return 1
        elif s<=15: return 2
        else: return 3
    df["PSQI_SEVERITY"] = df["PSQI_TOT"].apply(sev)
    counts = df["PSQI_SEVERITY"].value_counts()
    valid = counts[counts >= 2].index
    df = df[df["PSQI_SEVERITY"].isin(valid)].reset_index(drop=True)
    feats = ["PSQIDURAT","PSQILATEN","PSQIHSE","PSQIDISTB",
             "qx_psqi_06","qx_psqi_07","PSQIDAYDYS"]
    X = df[feats].fillna(df[feats].median()); y = df["PSQI_SEVERITY"]
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); Xtr_s = pd.DataFrame(sc.fit_transform(Xtr), columns=feats)
    Xte_s = pd.DataFrame(sc.transform(Xte), columns=feats)
    m = RandomForestClassifier(n_estimators=200, max_depth=6,
                               random_state=42, class_weight="balanced")
    m.fit(Xtr_s, ytr)
    sev_map = {0:"Good Sleep",1:"Mild Poor",2:"Moderate Poor",3:"Severe Poor"}
    unique = sorted(y.unique())
    names = [sev_map.get(l, f"Class {l}") for l in unique]
    return m, Xtr_s, Xte_s, ytr, yte, names, feats, pd.DataFrame(sc.transform(X), columns=feats), y

def _load_and_train_pss():
    df = pd.read_csv(os.path.join(BASE, "dataset","model","models","pss","pss.csv"))
    def sev(s):
        if s<=13: return 0
        elif s<=26: return 1
        else: return 2
    df["severity"] = df["score"].apply(sev)
    qc = [f"question{i}" for i in range(1,15)]
    tc = [f"time{i}" for i in range(1,15)]
    df["avg_rt"] = df[tc].mean(axis=1); df["max_rt"] = df[tc].max(axis=1)
    feats = qc + ["avg_rt","max_rt"]
    X = df[feats]; y = df["severity"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = RandomForestClassifier(n_estimators=400, random_state=42,
                               class_weight="balanced", n_jobs=-1)
    m.fit(Xtr, ytr)
    names = ["Low Stress","Moderate Stress","High Stress"]
    return m, Xtr, Xte, ytr, yte, names, feats, X, y


# ══════════════════════════════════════════════════════════
# CHART GENERATORS
# ══════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────
# 1. CONFUSION MATRIX HEATMAPS
# ──────────────────────────────────────────────────────────
def chart_confusion_matrices(all_data):
    for key, d in all_data.items():
        y_pred = d["model"].predict(d["Xte"])
        cm = confusion_matrix(d["yte"], y_pred)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=d["class_names"], yticklabels=d["class_names"],
                    linewidths=0.5, linecolor=GRID_CLR, cbar_kws={"shrink": 0.8},
                    ax=ax)
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label",      fontsize=12, fontweight="bold")
        ax.set_title(f"Confusion Matrix — {key}", fontsize=14,
                     fontweight="bold", pad=15)
        plt.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, f"cm_{key.lower().replace(' ','_').replace('-','')}.png"),
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
    print("  ✅ Confusion matrix heatmaps saved")


# ──────────────────────────────────────────────────────────
# 2. PER-MODEL PRECISION / RECALL / F1 BAR CHARTS
# ──────────────────────────────────────────────────────────
def chart_per_model_prf(all_data):
    for key, d in all_data.items():
        y_pred = d["model"].predict(d["Xte"])
        report = classification_report(d["yte"], y_pred,
                                       target_names=d["class_names"],
                                       output_dict=True)
        classes = d["class_names"]
        prec = [report[c]["precision"] for c in classes]
        rec  = [report[c]["recall"]    for c in classes]
        f1s  = [report[c]["f1-score"]  for c in classes]

        x = np.arange(len(classes))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(classes)*2), 5.5))
        bars1 = ax.bar(x - w, prec, w, label="Precision", color=ACCENT[0],
                       edgecolor=DARK_BG, linewidth=0.5, zorder=3)
        bars2 = ax.bar(x,     rec,  w, label="Recall",    color=ACCENT[1],
                       edgecolor=DARK_BG, linewidth=0.5, zorder=3)
        bars3 = ax.bar(x + w, f1s,  w, label="F1 Score",  color=ACCENT[2],
                       edgecolor=DARK_BG, linewidth=0.5, zorder=3)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8,
                        color=TEXT_CLR)

        ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=12); ax.set_xlabel("Class", fontsize=12)
        ax.set_title(f"Precision / Recall / F1 — {key}", fontsize=14,
                     fontweight="bold", pad=15)
        ax.legend(loc="upper right", framealpha=0.8, facecolor=CARD_BG,
                  edgecolor=GRID_CLR)
        plt.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, f"prf_{key.lower().replace(' ','_').replace('-','')}.png"),
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
    print("  ✅ Precision / Recall / F1 charts saved")


# ──────────────────────────────────────────────────────────
# 3. OVERALL ACCURACY COMPARISON
# ──────────────────────────────────────────────────────────
def chart_accuracy_comparison(all_data):
    names, accs = [], []
    for key, d in all_data.items():
        y_pred = d["model"].predict(d["Xte"])
        accs.append(accuracy_score(d["yte"], y_pred))
        names.append(key)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(names, accs, color=ACCENT[:len(names)],
                   edgecolor=DARK_BG, linewidth=0.5, height=0.55, zorder=3)
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.005, bar.get_y() + bar.get_height()/2,
                f"{acc:.2%}", va="center", fontsize=11, fontweight="bold",
                color=TEXT_CLR)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=16,
                 fontweight="bold", pad=15)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "accuracy_comparison.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ Accuracy comparison chart saved")


# ──────────────────────────────────────────────────────────
# 4. ROC CURVES (One-vs-Rest for multiclass)
# ──────────────────────────────────────────────────────────
def chart_roc_curves(all_data):
    for key, d in all_data.items():
        model = d["model"]; Xte = d["Xte"]; yte = np.array(d["yte"])
        classes = sorted(np.unique(yte))
        n_classes = len(classes)

        # Get probability estimates
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(Xte)
        else:
            continue

        yte_bin = label_binarize(yte, classes=classes)
        if n_classes == 2:
            yte_bin = np.hstack([1-yte_bin, yte_bin])

        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ACCENT[:n_classes]
        for i, (cls, color) in enumerate(zip(classes, colors)):
            fpr, tpr, _ = roc_curve(yte_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            label_name = d["class_names"][i] if i < len(d["class_names"]) else f"Class {cls}"
            ax.plot(fpr, tpr, color=color, lw=2.2,
                    label=f"{label_name} (AUC={roc_auc:.3f})")

        ax.plot([0,1],[0,1], "w--", alpha=0.3, lw=1)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate",  fontsize=12)
        ax.set_title(f"ROC Curves — {key}", fontsize=14,
                     fontweight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.8,
                  facecolor=CARD_BG, edgecolor=GRID_CLR)
        ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
        plt.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, f"roc_{key.lower().replace(' ','_').replace('-','')}.png"),
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
    print("  ✅ ROC curve charts saved")


# ──────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE (Random Forest models only)
# ──────────────────────────────────────────────────────────
def chart_feature_importance(all_data):
    for key, d in all_data.items():
        model = d["model"]
        if not hasattr(model, "feature_importances_"):
            continue
        imp = model.feature_importances_
        feats = d["features"]
        idx = np.argsort(imp)

        fig, ax = plt.subplots(figsize=(8, max(4, len(feats)*0.4)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feats)))
        ax.barh(np.array(feats)[idx], imp[idx], color=colors,
                edgecolor=DARK_BG, linewidth=0.5, zorder=3)
        for i, (v, f) in enumerate(zip(imp[idx], np.array(feats)[idx])):
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9,
                    color=TEXT_CLR)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"Feature Importance — {key}", fontsize=14,
                     fontweight="bold", pad=15)
        plt.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, f"feat_{key.lower().replace(' ','_').replace('-','')}.png"),
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
    print("  ✅ Feature importance charts saved")


# ──────────────────────────────────────────────────────────
# 6. CLASS DISTRIBUTION DONUT CHARTS
# ──────────────────────────────────────────────────────────
def chart_class_distributions(all_data):
    for key, d in all_data.items():
        y_full = np.array(d["y_full"])
        classes = sorted(np.unique(y_full))
        counts = [np.sum(y_full == c) for c in classes]
        labels = d["class_names"]

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        colors = ACCENT[:len(classes)]
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=140,
            pctdistance=0.78, wedgeprops=dict(width=0.45, edgecolor=DARK_BG, linewidth=2)
        )
        for t in texts: t.set_color(TEXT_CLR); t.set_fontsize(10)
        for t in autotexts: t.set_color("white"); t.set_fontsize(9); t.set_fontweight("bold")

        centre_circle = plt.Circle((0,0), 0.55, fc=DARK_BG)
        ax.add_artist(centre_circle)
        ax.text(0, 0, f"n={sum(counts)}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=TEXT_CLR)
        ax.set_title(f"Class Distribution — {key}", fontsize=14,
                     fontweight="bold", pad=15)
        plt.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, f"dist_{key.lower().replace(' ','_').replace('-','')}.png"),
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
    print("  ✅ Class distribution charts saved")


# ──────────────────────────────────────────────────────────
# 7. RADAR / SPIDER CHART — ALL MODELS
# ──────────────────────────────────────────────────────────
def chart_radar(all_data):
    metrics_map = {}
    for key, d in all_data.items():
        y_pred = d["model"].predict(d["Xte"])
        yte = d["yte"]
        avg = "weighted"
        metrics_map[key] = {
            "Accuracy":  accuracy_score(yte, y_pred),
            "Precision": precision_score(yte, y_pred, average=avg, zero_division=0),
            "Recall":    recall_score(yte, y_pred, average=avg, zero_division=0),
            "F1 Score":  f1_score(yte, y_pred, average=avg, zero_division=0),
        }

    categories = list(list(metrics_map.values())[0].keys())
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(CARD_BG)
    ax.spines['polar'].set_color(GRID_CLR)

    for i, (key, metrics) in enumerate(metrics_map.items()):
        vals = [metrics[c] for c in categories] + [metrics[categories[0]]]
        ax.plot(angles, vals, 'o-', linewidth=2, label=key,
                color=ACCENT[i % len(ACCENT)], markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=ACCENT[i % len(ACCENT)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Radar", fontsize=16,
                 fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9,
              framealpha=0.8, facecolor=CARD_BG, edgecolor=GRID_CLR)
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "radar_comparison.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ Radar chart saved")


# ──────────────────────────────────────────────────────────
# 8. METRICS HEATMAP (Models × Metrics)
# ──────────────────────────────────────────────────────────
def chart_metrics_heatmap(all_data):
    rows = []
    for key, d in all_data.items():
        y_pred = d["model"].predict(d["Xte"])
        yte = d["yte"]
        avg = "weighted"
        rows.append({
            "Model":     key,
            "Accuracy":  accuracy_score(yte, y_pred),
            "Precision": precision_score(yte, y_pred, average=avg, zero_division=0),
            "Recall":    recall_score(yte, y_pred, average=avg, zero_division=0),
            "F1 Score":  f1_score(yte, y_pred, average=avg, zero_division=0),
        })
    df = pd.DataFrame(rows).set_index("Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="YlGnBu",
                linewidths=0.8, linecolor=GRID_CLR, cbar_kws={"shrink": 0.8},
                vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("All Models — Performance Metrics Heatmap", fontsize=14,
                 fontweight="bold", pad=15)
    ax.set_ylabel(""); ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "metrics_heatmap.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ Metrics heatmap saved")


# ──────────────────────────────────────────────────────────
# 9. LEARNING CURVES (representative models)
# ──────────────────────────────────────────────────────────
def chart_learning_curves(all_data):
    # Pick up to 3 representative models
    picks = list(all_data.keys())[:3]
    for key in picks:
        d = all_data[key]
        X_full = d["X_full"]; y_full = d["y_full"]
        model_clone = type(d["model"])(**d["model"].get_params())

        sizes, train_sc, val_sc = learning_curve(
            model_clone, X_full, y_full,
            cv=5, scoring="accuracy",
            train_sizes=np.linspace(0.2, 1.0, 6),
            random_state=42, n_jobs=-1
        )

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.fill_between(sizes, train_sc.mean(axis=1)-train_sc.std(axis=1),
                        train_sc.mean(axis=1)+train_sc.std(axis=1),
                        alpha=0.15, color=ACCENT[0])
        ax.fill_between(sizes, val_sc.mean(axis=1)-val_sc.std(axis=1),
                        val_sc.mean(axis=1)+val_sc.std(axis=1),
                        alpha=0.15, color=ACCENT[1])
        ax.plot(sizes, train_sc.mean(axis=1), 'o-', color=ACCENT[0],
                lw=2.2, label="Training Score")
        ax.plot(sizes, val_sc.mean(axis=1), 's-', color=ACCENT[1],
                lw=2.2, label="Cross-Val Score")
        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Learning Curve — {key}", fontsize=14,
                     fontweight="bold", pad=15)
        ax.legend(loc="lower right", framealpha=0.8, facecolor=CARD_BG,
                  edgecolor=GRID_CLR)
        ax.set_ylim(0, 1.08)
        plt.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, f"lc_{key.lower().replace(' ','_').replace('-','')}.png"),
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
    print("  ✅ Learning curve charts saved")


# ──────────────────────────────────────────────────────────
# 10. DASHBOARD SUMMARY (multi-panel)
# ──────────────────────────────────────────────────────────
def chart_dashboard(all_data):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    names, accs, precs, recs, f1s_list = [], [], [], [], []
    for key, d in all_data.items():
        y_pred = d["model"].predict(d["Xte"])
        yte = d["yte"]; avg = "weighted"
        names.append(key)
        accs.append(accuracy_score(yte, y_pred))
        precs.append(precision_score(yte, y_pred, average=avg, zero_division=0))
        recs.append(recall_score(yte, y_pred, average=avg, zero_division=0))
        f1s_list.append(f1_score(yte, y_pred, average=avg, zero_division=0))

    # Panel 1: Accuracy bars
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(names, accs, color=ACCENT[:len(names)], height=0.55,
                    edgecolor=DARK_BG, zorder=3)
    for bar, a in zip(bars, accs):
        ax1.text(a+0.01, bar.get_y()+bar.get_height()/2,
                 f"{a:.1%}", va="center", fontsize=9, color=TEXT_CLR)
    ax1.set_xlim(0, 1.15); ax1.invert_yaxis()
    ax1.set_title("Accuracy", fontsize=13, fontweight="bold")

    # Panel 2: Precision bars
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.barh(names, precs, color=ACCENT[:len(names)], height=0.55,
                    edgecolor=DARK_BG, zorder=3)
    for bar, p in zip(bars, precs):
        ax2.text(p+0.01, bar.get_y()+bar.get_height()/2,
                 f"{p:.1%}", va="center", fontsize=9, color=TEXT_CLR)
    ax2.set_xlim(0, 1.15); ax2.invert_yaxis()
    ax2.set_title("Precision (weighted)", fontsize=13, fontweight="bold")

    # Panel 3: Recall bars
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.barh(names, recs, color=ACCENT[:len(names)], height=0.55,
                    edgecolor=DARK_BG, zorder=3)
    for bar, r in zip(bars, recs):
        ax3.text(r+0.01, bar.get_y()+bar.get_height()/2,
                 f"{r:.1%}", va="center", fontsize=9, color=TEXT_CLR)
    ax3.set_xlim(0, 1.15); ax3.invert_yaxis()
    ax3.set_title("Recall (weighted)", fontsize=13, fontweight="bold")

    # Panel 4: F1 Score bars
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.barh(names, f1s_list, color=ACCENT[:len(names)], height=0.55,
                    edgecolor=DARK_BG, zorder=3)
    for bar, f in zip(bars, f1s_list):
        ax4.text(f+0.01, bar.get_y()+bar.get_height()/2,
                 f"{f:.1%}", va="center", fontsize=9, color=TEXT_CLR)
    ax4.set_xlim(0, 1.15); ax4.invert_yaxis()
    ax4.set_title("F1 Score (weighted)", fontsize=13, fontweight="bold")

    # Panel 5: Grouped bar — all metrics side by side
    ax5 = fig.add_subplot(gs[1, 1:])
    x = np.arange(len(names)); w = 0.2
    ax5.bar(x - 1.5*w, accs,     w, label="Accuracy",  color=ACCENT[0], edgecolor=DARK_BG, zorder=3)
    ax5.bar(x - 0.5*w, precs,    w, label="Precision", color=ACCENT[1], edgecolor=DARK_BG, zorder=3)
    ax5.bar(x + 0.5*w, recs,     w, label="Recall",    color=ACCENT[2], edgecolor=DARK_BG, zorder=3)
    ax5.bar(x + 1.5*w, f1s_list, w, label="F1 Score",  color=ACCENT[3], edgecolor=DARK_BG, zorder=3)
    ax5.set_xticks(x); ax5.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax5.set_ylim(0, 1.12)
    ax5.set_title("All Metrics Comparison", fontsize=13, fontweight="bold")
    ax5.legend(loc="upper right", fontsize=9, framealpha=0.8,
               facecolor=CARD_BG, edgecolor=GRID_CLR)

    fig.suptitle("FibroTracker — Model Evaluation Dashboard", fontsize=18,
                 fontweight="bold", y=0.98)
    fig.savefig(os.path.join(CHART_DIR, "dashboard_summary.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ Dashboard summary saved")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "📊" * 35)
    print("  FIBROTRACKER – CHART GENERATION FOR DOCUMENTATION")
    print("📊" * 35 + "\n")

    print("  Loading datasets & training models …")
    loaders = {
        "Fibro Risk":     _load_and_train_fibro,
        "FSS Fatigue":    _load_and_train_fss,
        "GAD-7 Anxiety":  _load_and_train_gad,
        "PHQ-9 Depression": _load_and_train_phq,
        "PSQI Sleep":     _load_and_train_psqi,
        "PSS-14 Stress":  _load_and_train_pss,
    }

    all_data = {}
    for name, loader in loaders.items():
        print(f"    ▸ {name}")
        m, Xtr, Xte, ytr, yte, class_names, feats, X_full, y_full = loader()
        all_data[name] = {
            "model": m, "Xtr": Xtr, "Xte": Xte,
            "ytr": ytr, "yte": yte,
            "class_names": class_names,
            "features": feats,
            "X_full": X_full, "y_full": y_full,
        }

    print(f"\n  All models ready. Generating charts → {CHART_DIR}\n")

    chart_confusion_matrices(all_data)
    chart_per_model_prf(all_data)
    chart_accuracy_comparison(all_data)
    chart_roc_curves(all_data)
    chart_feature_importance(all_data)
    chart_class_distributions(all_data)
    chart_radar(all_data)
    chart_metrics_heatmap(all_data)
    chart_learning_curves(all_data)
    chart_dashboard(all_data)

    total = len([f for f in os.listdir(CHART_DIR) if f.endswith(".png")])
    print(f"\n  🎉 Done! {total} charts saved to '{CHART_DIR}'")
    print("  Charts ready for documentation.\n")
