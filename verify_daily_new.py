import requests
import json
import datetime

BASE_URL = "http://127.0.0.1:5000"

def test_daily_entry():
    session = requests.Session()

    # 1. Login/Register
    username = f"test_daily_{datetime.datetime.now().timestamp()}"
    password = "password123"
    
    # Register
    res = session.post(f"{BASE_URL}/api/register", json={
        "username": username, "email": f"{username}@example.com", 
        "password": password, "confirm_password": password
    })
    
    # Login
    res = session.post(f"{BASE_URL}/api/login", json={
        "username": username, "password": password
    })
    if res.status_code != 200:
        print("Login failed:", res.text)
        return

    # 2. Submit Daily Entry (New Format)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    payload = {
        "entry_date": today,
        "pain_score": 5,
        "fatigue_score": 6,
        "sleep_quality": 4,
        "cognitive_difficulty": 3,
        "stress_score": 2,
        "physical_activity_level": "Moderate",
        "sleep_duration_category": "5-6 hours",
        "weather_sensitivity_bool": 1,
        "sensory_sensitivity_score": 7,
        "recent_infection": 0,
        "menstrual_phase": "Not applicable",
        "pain_area_count": 4
    }
    
    print("Submitting daily entry...")
    res = session.post(f"{BASE_URL}/api/daily-entry", json=payload)
    print("Submit Status:", res.status_code)
    print("Submit Response:", res.text)
    
    if res.status_code != 200:
        print("FAIL: Submission error.")
        return

    # 3. Verify Data Saved
    print("Verifying saved data...")
    res = session.get(f"{BASE_URL}/api/daily-entry", params={"date": today})
    if res.status_code == 200:
        data = res.json()
        print("Retrieved Data:", json.dumps(data, indent=2))
        
        # Check new fields
        if data.get("physical_activity_level") == "Moderate":
            print("PASS: physical_activity_level saved.")
        else:
            print("FAIL: physical_activity_level mismatch.")
            
        if data.get("weather_sensitivity_bool") in [1, True]:
             print("PASS: weather_sensitivity_bool saved.")
        else:
             print("FAIL: weather_sensitivity_bool mismatch.")
             
        if data.get("cognitive_difficulty") == 3:
             print("PASS: cognitive_difficulty saved.")
        else:
             print("FAIL: cognitive_difficulty mismatch.")

    else:
        print("FAIL: Could not retrieve entry:", res.text)

if __name__ == "__main__":
    try:
        test_daily_entry()
    except Exception as e:
        print("Error during test:", e)
