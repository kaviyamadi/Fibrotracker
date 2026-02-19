import requests
import json

BASE_URL = "http://127.0.0.1:5000"
SESSION = requests.Session()

def register_and_login():
    # Register
    username = "test_user_screening"
    password = "password123"
    email = "test@example.com"
    
    print("Registering...")
    res = SESSION.post(f"{BASE_URL}/api/register", json={
        "username": username,
        "password": password,
        "confirm_password": password,
        "email": email
    })
    
    if res.status_code == 409:
        print("User already exists, logging in...")
    elif res.status_code != 200:
        print(f"Registration failed: {res.text}")
        return False

    # Login
    print("Logging in...")
    res = SESSION.post(f"{BASE_URL}/api/login", json={
        "username": username,
        "password": password
    })
    
    if res.status_code == 200:
        print("Login successful")
        return True
    else:
        print(f"Login failed: {res.text}")
        return False

def test_screening_submission():
    payload = {
        "first_answers": {"f1": True, "f2": True, "f3": True, "f4": False, "f5": False, "f6": True},
        "wpi_regions": ["L jaw", "Neck", "Upper back", "Lower back", "Chest", "Abdomen", "L upper leg"], # 7 regions
        "sss_answers": {"fatigue": 3, "sleep": 2, "cognitive": 2}, # A = 7
        "sss_somatic": {"headache": 1, "abdomenPain": 1, "depression": 0}, # B = 2
        # SSS Total = 9
        # WPI = 7
        # Rule 1 Met: WPI >= 7 & SSS >= 5. Primary Score = 1.0.
        
        "secondary_symptoms": ["secondary_headache", "secondary_ibs", "secondary_depression"], # 3 items
        # Secondary Score: 3/10 = 0.3
        
        "risk_factors": {"r1": True, "r5": True}, # Family History + Anxiety = 0.50
        # Risk Score: 0.50 / 1.75 = 0.2857
        
        "duration_4_weeks": True
        # Rule 3 Met.
        
        # Total Risk Score Calculation:
        # 0.6 * 1.0 (Primary) = 0.6
        # 0.3 * 0.3 (Secondary) = 0.09
        # 0.1 * 0.2857 (Risk) = 0.028
        # Total = 0.6 + 0.09 + 0.028 = 0.718
        # Category Should be "High" (>= 0.7)
    }
    
    print("Submitting screening...")
    res = SESSION.post(f"{BASE_URL}/api/screening", json=payload)
    
    if res.status_code == 200:
        data = res.json()
        print(f"Submission success: {data}")
        res_data = data['result']
        
        # Check Risk Probability
        prob = res_data.get('risk_probability')
        print(f"Risk Probability: {prob}")
        if 0.7 <= prob <= 0.75:
            print("PASS: Risk probability within expected range (~0.718).")
        else:
            print(f"FAIL: Risk probability {prob} unexpected.")
            
        # Check Category
        if res_data.get('risk_level') == "High":
            print("PASS: Risk Category is High.")
        else:
            print(f"FAIL: Risk Category mismatch. Got {res_data.get('risk_level')}")
            
        # Check Eligibility
        if res_data.get('is_eligible') == True:
             print("PASS: Eligibility is True.")
        else:
             print("FAIL: Eligibility is False.")
             
    else:
        print(f"Submission failed: {res.text}")

def check_profile_status():
    print("Checking latest screening functionality...")
    res = SESSION.get(f"{BASE_URL}/api/latest-screening")
    
    if res.status_code == 200:
        data = res.json()
        print(f"Latest Screening Data: {data}")
        if data.get('risk_level') == "High":
             print("PASS: Profile API reflects high risk.")
        else:
             print("FAIL: Profile API incorrect.")

        # Check FiRST Score
        # Answered 4 True in payload
        if data.get('first_score') == 4:
            print("PASS: FiRST score is 4.")
        else:
            print(f"FAIL: FiRST score incorrect. Got {data.get('first_score')}")

        # Check Duration
        # Answered True for duration_4_weeks -> "more_than_3_months"
        if data.get('duration') == "more_than_3_months":
            print("PASS: Duration is 'more_than_3_months'.")
        else:
            print(f"FAIL: Duration incorrect. Got {data.get('duration')}")
    else:
        print(f"Failed to get latest screening: {res.text}")

if __name__ == "__main__":
    if register_and_login():
        test_screening_submission()
        check_profile_status()
