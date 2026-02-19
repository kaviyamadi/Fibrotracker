import requests
import json
import random

BASE_URL = "http://localhost:5000"

# 1. Login (if needed, or Mock session)
# Since we don't have easy login via script without simulating browser, 
# we might need to rely on the fact that we can't easily test auth-protected routes 
# from a standalone script without a valid session cookie.
# However, for verification of *logic*, we can import app and test functions directly?
# No, `app` imports `genai` and DB, might be complex to context-switch.
# Better to use the running server if it was running. 
# But I can't guarantee server is running.
# I will create a unit-test style script that imports `app` and mocks request context.

import sys
import os
sys.path.append(os.getcwd())

# Mock environment variables to avoid errors during import
os.environ["GEMINI_API_KEY"] = "dummy"

try:
    from app import app, SCREENING_MODEL, GAD_MODEL, PHQ_MODEL
    print("✅ App imported successfully.")
except ImportError as e:
    print(f"⚠️ Could not import app. Error: {e}")
    exit(1)

def test_model_loading():
    if SCREENING_MODEL:
        print("✅ Screening model loaded.")
    else:
        print("❌ Screening model NOT loaded.")

    try:
        from app import GAD_MODEL, PHQ_MODEL
        if GAD_MODEL: print("✅ GAD model loaded.")
        else: print("❌ GAD model NOT loaded.")
        
        if PHQ_MODEL: print("✅ PHQ model loaded.")
        else: print("❌ PHQ model NOT loaded.")
    except ImportError:
        print("⚠️ Could not access GAD/PHQ models from app module.")

def test_screening_logic():
    print("\ntesting screening logic...")
    # diverse dummy data
    data = {
        "phq9_data": {
            "question1": 2, "question2": 1, "question3": 2, "question4": 3,
            "question5": 0, "question6": 1, "question7": 2, "question8": 3, "question9": 0,
            "times": {f"time{i}": random.uniform(2.0, 10.0) for i in range(1, 10)}
        },
        "gad7_data": {
            "question1": 1, "question2": 2, "question3": 1, "question4": 0,
            "question5": 1, "question6": 2, "question7": 1,
            "times": {f"time{i}": random.uniform(2.0, 10.0) for i in range(1, 8)}
        }
    }
    
    # We can't easily invoke the route without DB user session.
    # But we can verify if models predict on this data.
    
    from app import GAD_MODEL, PHQ_MODEL
    import pandas as pd
    
    if GAD_MODEL:
        gad_input = {}
        for k, v in data['gad7_data'].items():
            if k.startswith('question'): gad_input[k] = v
        for k, v in data['gad7_data']['times'].items():
            gad_input[k] = v
            
        df = pd.DataFrame([gad_input])
        time_cols = [f'time{i}' for i in range(1, 8)]
        q_cols = [f'question{i}' for i in range(1, 8)]
        df["avg_response_time"] = df[time_cols].mean(axis=1)
        df["max_response_time"] = df[time_cols].max(axis=1)
        features = df[q_cols + ["avg_response_time", "max_response_time"]]
        
        pred = GAD_MODEL.predict(features)[0]
        print(f"✅ GAD Prediction: {pred} (0-3)")
        
    if PHQ_MODEL:
        phq_input = {}
        for k, v in data['phq9_data'].items():
            if k.startswith('question'): phq_input[k] = v
        for k, v in data['phq9_data']['times'].items():
            phq_input[k] = v
            
        df = pd.DataFrame([phq_input])
        time_cols = [f'time{i}' for i in range(1, 10)]
        q_cols = [f'question{i}' for i in range(1, 10)]
        df["avg_response_time"] = df[time_cols].mean(axis=1)
        df["max_response_time"] = df[time_cols].max(axis=1)
        features = df[q_cols + ["avg_response_time", "max_response_time"]]
        
        pred = PHQ_MODEL.predict(features)[0]
        print(f"✅ PHQ Prediction: {pred} (0-4)")

if __name__ == "__main__":
    test_model_loading()
    try:
        test_screening_logic()
    except Exception as e:
        print(f"❌ Verification Logic Error: {e}")
