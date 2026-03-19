import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

# Ignore K-Means warnings for demonstration purposes
warnings.filterwarnings('ignore')

def generate_synthetic_data(days=60, scenario="fluctuating"):
    """
    Generate synthetic daily symptom inputs mimicking real patient behavior.
    """
    dates = [datetime.today() - timedelta(days=x) for x in range(days)]
    dates.reverse()

    data = []
    
    # Base states
    pain, fatigue, sleep, stress = 4, 5, 6, 4
    
    for i, date in enumerate(dates):
        if scenario == "stable_low":
            pain = np.random.normal(2, 0.5)
            fatigue = np.random.normal(3, 0.5)
            sleep = np.random.normal(7, 0.5) # Higher sleep = better
            stress = np.random.normal(2, 0.5)
        elif scenario == "increasing_trend":
            pain = 2 + (i / days) * 6 + np.random.normal(0, 0.5) # Gradually goes from 2 to 8
            fatigue = 3 + (i / days) * 5 + np.random.normal(0, 0.5)
            sleep = 8 - (i / days) * 4 + np.random.normal(0, 0.5)
            stress = 3 + (i / days) * 4 + np.random.normal(0, 0.5)
        elif scenario == "trigger_spikes":
            # Mostly stable, but spikes when stress goes high
            stress = np.random.choice([2, 3, 8, 9], p=[0.4, 0.4, 0.1, 0.1])
            if stress >= 8:
                pain = np.random.normal(8, 0.5)
                fatigue = np.random.normal(8, 0.5)
                sleep = np.random.normal(3, 0.5)
            else:
                pain = np.random.normal(4, 0.5)
                fatigue = np.random.normal(4, 0.5)
                sleep = np.random.normal(7, 0.5)
        elif scenario == "persistent_high":
            # High pain consistently to trigger persistent risk and K-Means
            pain = np.random.normal(8, 0.5)
            fatigue = np.random.normal(8, 0.5)
            sleep = np.random.normal(3, 0.5)
            stress = np.random.choice([4, 9], p=[0.3, 0.7]) # Mostly high stress

        else:
            # Random fluctuations
            pain += np.random.normal(0, 1)
            fatigue += np.random.normal(0, 1)
            sleep += np.random.normal(0, 1)
            stress += np.random.normal(0, 1)

        # Introduce missing data (~10% chance)
        if np.random.rand() < 0.1:
            pain = np.nan
        
        # Clip to bounds 0-10
        data.append({
            "date": date.strftime('%Y-%m-%d'),
            "pain_score": np.clip(pain, 0, 10) if not np.isnan(pain) else np.nan,
            "fatigue_score": np.clip(fatigue, 0, 10),
            "sleep_quality": np.clip(sleep, 0, 10),
            "stress_level": np.clip(stress, 0, 10),
            "cognitive_difficulty": np.clip(np.random.normal(4, 1), 0, 10)
        })

    df = pd.DataFrame(data)
    return df

def apply_validation_and_locf(df):
    """
    Handle missing values using LOCF (Last Observation Carried Forward).
    """
    print("\n--- 1. Data Validation & Missing Data Handling ---")
    missing_count = df['pain_score'].isna().sum()
    print(f"Detected {missing_count} missing pain score entries. Applying LOCF...")
    
    # LOCF Imputation
    df_clean = df.copy()
    df_clean['pain_score'] = df_clean['pain_score'].ffill().bfill() # bfill handles if first day is NaN
    
    return df_clean

def compute_metrics(df):
    """
    Compute key metrics: mean pain, variability, trend slope, flare count.
    Assumes df represents a specific period (e.g. 1 week).
    """
    mean_pain = df['pain_score'].mean()
    variability = df['pain_score'].std()
    
    # Trend slope (simplistic start vs end of week or linear regression slope)
    if len(df) > 1:
        x = np.arange(len(df))
        y = df['pain_score'].values
        slope, _ = np.polyfit(x, y, 1)
    else:
        slope = 0
        
    # Flare count (threshold: pain >= 7)
    flare_count = (df['pain_score'] >= 7).sum()
    
    return mean_pain, variability, slope, flare_count

def assign_risk_level(mean_pain, flare_count):
    if mean_pain > 7 or flare_count >= 3:
        return 'High'
    elif mean_pain > 4 or flare_count >= 1:
        return 'Moderate'
    else:
        return 'Low'

def execute_clustering(df):
    """
    Execute K-Means on symptoms and triggers to identify dominant patterns.
    """
    print("\n--- 4. Target Trigger Clustering (K-Means) ---")
    print("Persistent risk detected. Running K-Means to identify dominant triggers...")
    
    features = ['pain_score', 'fatigue_score', 'sleep_quality', 'stress_level']
    X = df[features].values
    
    if len(X) < 2:
        print("Not enough data to cluster.")
        return "Unknown"
        
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    # --- EVALUATE K-MEANS ----
    try:
        sil_score = silhouette_score(X, df['cluster'])
        ch_score = calinski_harabasz_score(X, df['cluster'])
        db_score = davies_bouldin_score(X, df['cluster'])
        print("\n  [K-Means Clustering Evaluation]")
        print(f"  - Silhouette Score        : {sil_score:.4f}  (> 0.5 is good, closer to 1 is better)")
        print(f"  - Calinski-Harabasz Score : {ch_score:.2f}  (Higher is better)")
        print(f"  - Davies-Bouldin Score    : {db_score:.4f}  (Lower is better, ideally < 1.0)")
    except Exception as e:
        print(f"\n  [K-Means Evaluation Skipped: {e}]")
    
    # Find the cluster with highest average pain
    cluster_means = df.groupby('cluster')[features].mean()
    high_pain_cluster = cluster_means['pain_score'].idxmax()
    
    trigger_profile = cluster_means.loc[high_pain_cluster]
    print("\nCluster Profiling for High-Pain periods:")
    print(trigger_profile.to_string())
    
    # Identify dominant trigger by checking which trigger (sleep or stress) deviates most in the high pain cluster
    # (Simplified logic: if stress is high, it's stress. If sleep is low, it's sleep)
    dominant_trigger = "Various"
    if trigger_profile['stress_level'] >= 7:
        dominant_trigger = "Stress-related"
    elif trigger_profile['sleep_quality'] <= 4:
        dominant_trigger = "Sleep-deprivation"
        
    print(f"\n👉 Result: Identified Dominant Trigger = {dominant_trigger}")
    return dominant_trigger

def evaluate_tracking_pipeline(scenario="trigger_spikes", days=60):
    print("=" * 60)
    print(f"EVALUATING TRACKING PIPELINE: SCENARIO => {scenario.upper()} ({days} days)")
    print("=" * 60)
    
    # Generate Data
    df_raw = generate_synthetic_data(days=days, scenario=scenario)
    
    # Step 1: LOCF
    df_clean = apply_validation_and_locf(df_raw)
    
    # Step 2 & 3: Weekly Metrics & Risk Assessment
    print("\n--- 2. Weekly Metrics & Risk Assessment ---")
    weeks = [df_clean.iloc[i:i+7] for i in range(0, len(df_clean), 7)]
    
    consecutive_high_risk = 0
    stable_low_weeks = 0
    persistent_risk_flag = False
    tracking_active = True
    
    for week_num, week_df in enumerate(weeks, 1):
        if len(week_df) == 0: continue
            
        mean_pain, var, slope, flares = compute_metrics(week_df)
        risk = assign_risk_level(mean_pain, flares)
        
        print(f"Week {week_num}: Mean Pain={mean_pain:.1f}, Variability={var:.1f}, Trend={slope:.2f}, Flares={flares} => Risk: {risk}")
        
        # Persistence Logic
        if risk == 'High':
            consecutive_high_risk += 1
            stable_low_weeks = 0
        elif risk == 'Low':
            stable_low_weeks += 1
            consecutive_high_risk = 0
        else:
            consecutive_high_risk = 0
            stable_low_weeks = 0
            
        if consecutive_high_risk >= 2:
            persistent_risk_flag = True
            
        # Adaptive tracking logic evaluated per week
        if stable_low_weeks >= 3: # E.g., Stop tracking after 3 stable low weeks
            tracking_active = False

    print("\n--- 3. Persistence Logic ---")
    if persistent_risk_flag:
        print("Consecutive high-risk weeks detected. Persistent Risk Flag set to TRUE.")
        dominant_trigger = execute_clustering(df_clean)
    else:
        print("No persistent risk detected.")
        
    print("\n--- 5. Adaptive Tracking Control ---")
    if tracking_active:
        print("Patient remains at Moderate/High risk or hasn't stabilized. Tracking remains ACTIVE.")
    else:
        print(f"Patient has achieved {stable_low_weeks} consecutive Low-Risk weeks. Outputting Tracking Control: STOP TRACKING.")
        
    print("\nPipeline execution completed successfully.\n")

if __name__ == "__main__":
    # Test multiple scenarios
    evaluate_tracking_pipeline("stable_low", days=35)
    evaluate_tracking_pipeline("increasing_trend", days=42)
    evaluate_tracking_pipeline("trigger_spikes", days=60)
    evaluate_tracking_pipeline("persistent_high", days=35)
