import random
import numpy as np
import pandas as pd

NUM_USERS = 2000
random.seed(42)
np.random.seed(42)

# -------------------------------
# Helper functions
# -------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def risk_category(prob):
    if prob <= 0.3:
        return "Low"
    elif prob <= 0.6:
        return "Moderate"
    return "High"

# -------------------------------
# 1. user_profile
# -------------------------------

users = []
for uid in range(1, NUM_USERS + 1):
    sex = random.choices(["Male", "Female"], weights=[0.35, 0.65])[0]
    users.append({
        "user_id": uid,
        "sex": sex,
        "age_group": random.choice(
            ["18-25","26-35","36-45","46-55","56-65","65+"]),
        "family_history": random.choice([0,1]),
        "weather_sensitivity": random.choice([0,1,2,3])  # None–High
    })

user_df = pd.DataFrame(users)

# -------------------------------
# 2. Primary Features (Rules)
# -------------------------------

primary_rows = []

for uid in range(1, NUM_USERS + 1):
    WPI = random.randint(0, 8)
    SSS = random.randint(0, 8)
    pain_regions = random.randint(0, 4)
    persistence_weeks = random.randint(1, 12)
    persistent_pain = int(persistence_weeks >= 4)

    # Rule 1
    rule1 = int(
        (2 <= WPI <= 3 and 4 <= SSS <= 5) or
        (WPI >= 4 and SSS >= 4) or
        (SSS >= 6 and persistent_pain)
    )

    # Rule 2
    rule2 = int(pain_regions >= 2)

    # Rule 3
    rule3 = persistent_pain

    primary_score = int(rule1 or rule2 or rule3)
    primary_scaled = (rule1 + rule2 + rule3) / 3

    primary_rows.append({
        "user_id": uid,
        "WPI": WPI,
        "SSS": SSS,
        "pain_regions": pain_regions,
        "symptom_persistence": persistence_weeks,
        "rule1": rule1,
        "rule2": rule2,
        "rule3": rule3,
        "primary_score": primary_score,
        "primary_scaled": primary_scaled
    })

primary_df = pd.DataFrame(primary_rows)

# -------------------------------
# 3. Secondary Symptoms
# -------------------------------

secondary_rows = []
for uid in range(1, NUM_USERS + 1):
    symptoms = np.random.binomial(1, 0.35, 10)
    total = symptoms.sum()
    secondary_norm = total / 10

    secondary_rows.append({
        "user_id": uid,
        "secondary_total": total,
        "secondary_score_norm": secondary_norm
    })

secondary_df = pd.DataFrame(secondary_rows)

# -------------------------------
# 4. Risk Factors
# -------------------------------

risk_rows = []
for uid in range(1, NUM_USERS + 1):
    factors = np.random.binomial(1, 0.25, 7)
    subtotal = factors.sum() * 0.25
    risk_fraction = subtotal / 1.75

    risk_rows.append({
        "user_id": uid,
        "risk_factor_fraction": risk_fraction,
        "rf_total": factors.sum()
    })

risk_df = pd.DataFrame(risk_rows)

# -------------------------------
# 5. Weighted Total Score
# -------------------------------

merged = primary_df.merge(
    secondary_df, on="user_id"
).merge(
    risk_df, on="user_id"
)

merged["modular_total_score"] = (
    0.6 * merged["primary_scaled"] +
    0.3 * merged["secondary_score_norm"] +
    0.1 * merged["risk_factor_fraction"]
)

# -------------------------------
# 6. Logistic Regression (Synthetic)
# -------------------------------

# Synthetic coefficients (for academic use)
logit = (
    3.2 * merged["modular_total_score"] +
    0.15 * merged["WPI"] +
    0.2 * merged["SSS"] +
    0.1 * merged["risk_factor_fraction"] -
    2.0
)

merged["risk_probability"] = sigmoid(logit)

# Rule override
merged["risk_category"] = merged.apply(
    lambda x: "High"
    if x["modular_total_score"] >= 0.7
    else risk_category(x["risk_probability"]),
    axis=1
)

# -------------------------------
# 7. Save Outputs
# -------------------------------

user_df.to_csv("synthetic_users.csv", index=False)
primary_df.to_csv("primary_features.csv", index=False)
secondary_df.to_csv("secondary_symptoms.csv", index=False)
risk_df.to_csv("risk_factors.csv", index=False)

merged[[
    "user_id",
    "modular_total_score",
    "risk_probability",
    "risk_category"
]].to_csv("screening_results.csv", index=False)

# ML-ready dataset
merged.to_csv("ml_training_dataset.csv", index=False)

print("✅ Synthetic dataset generated successfully (2000 users)")
