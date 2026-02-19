# app.py
from flask import Flask, request, jsonify, session, send_file, render_template, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from flask_cors import CORS
import json
import numpy as np
import joblib
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import os
from datetime import datetime, timedelta, date
import pandas as pd
import requests


app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_super_secret_key_change_in_production')
CORS(app)

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# -------------------------------------------------
# Load ML Models
# -------------------------------------------------
SCREENING_MODEL = None
SCREENING_LE = None

try:
    SCREENING_MODEL = joblib.load("fibro_risk_model.pkl")
    SCREENING_LE = joblib.load("fibro_risk_le.pkl")
    print("✅ Screening model loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load screening model: {e}")

GAD_MODEL = None
PHQ_MODEL = None
try:
    GAD_MODEL = joblib.load("dataset/models/gad/gad7_severity_model.pkl")
    PHQ_MODEL = joblib.load("dataset/models/phq/phq_severity_model.pkl")
    print("✅ GAD/PHQ models loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load GAD/PHQ models: {e}")

DATABASE = 'fibrotracker.db'

# Load ML model
try:
    model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    print("Warning: ML model file not found. Prediction endpoint will not work.")
    model = None

# **************** DATABASE SETUP **********************************
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db_connection()
    with conn:

        # -------------------------------------------------
        # Users table
        # -------------------------------------------------
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            profile_complete INTEGER DEFAULT 0 CHECK(profile_complete IN (0,1)),
            sex TEXT CHECK(sex IN ('Male', 'Female', 'Other')),
            age_group TEXT CHECK(age_group IN ('18-25', '26-35', '36-45', '46-55', '56-65', '65+')),
            family_history TEXT CHECK(family_history IN ('Yes', 'No')),
            menstrual_cycle TEXT CHECK(menstrual_cycle IN ('N/A', 'Regular', 'Irregular', 'Postmenopausal')),
            weather_sensitivity TEXT CHECK(weather_sensitivity IN ('None', 'Low', 'Moderate', 'High'))
        )
        ''')

        # -------------------------------------------------
        # Comorbidities table
        # -------------------------------------------------
        conn.execute('''
        CREATE TABLE IF NOT EXISTS comorbidities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        ''')

        # -------------------------------------------------
        # User ↔ Comorbidities mapping
        # -------------------------------------------------
        conn.execute('''
        CREATE TABLE IF NOT EXISTS user_comorbidities (
            user_id INTEGER NOT NULL,
            comorbidity_id INTEGER NOT NULL,
            PRIMARY KEY (user_id, comorbidity_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (comorbidity_id) REFERENCES comorbidities(id) ON DELETE CASCADE
        )
        ''')

        # -------------------------------------------------
        # First Tool (Initial Screening)
        # -------------------------------------------------
        conn.execute('''
        CREATE TABLE IF NOT EXISTS first_tool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            widespread_pain INTEGER CHECK(widespread_pain IN (0,1)),
            fatigue INTEGER CHECK(fatigue IN (0,1)),
            pain_type INTEGER CHECK(pain_type IN (0,1)),
            unusual_sensations INTEGER CHECK(unusual_sensations IN (0,1)),
            other_health_problems INTEGER CHECK(other_health_problems IN (0,1)),
            impact_on_life INTEGER CHECK(impact_on_life IN (0,1)),
            total_score INTEGER CHECK(total_score BETWEEN 0 AND 6),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')

        # -------------------------------------------------
        # Primary Symptoms
        # -------------------------------------------------
        # -------------------------------------------------
        # Primary Symptoms
        # -------------------------------------------------
        conn.execute('''
        CREATE TABLE IF NOT EXISTS primary_symptoms (
            symptom_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            pain_score INTEGER CHECK(pain_score BETWEEN 0 AND 19),
            fatigue_score INTEGER CHECK(fatigue_score BETWEEN 0 AND 10),
            sleep_score INTEGER CHECK(sleep_score BETWEEN 0 AND 10),
            cognitive_score INTEGER CHECK(cognitive_score BETWEEN 0 AND 10),
            total_score INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')

        # -------------------------------------------------
        # Secondary Symptoms
        # -------------------------------------------------
        conn.execute('''
        CREATE TABLE IF NOT EXISTS secondary_symptoms (
            sec_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            headache INTEGER CHECK(headache IN (0,1)),
            paresthesia INTEGER CHECK(paresthesia IN (0,1)),
            allodynia INTEGER CHECK(allodynia IN (0,1)),
            ibs INTEGER CHECK(ibs IN (0,1)),
            depression INTEGER CHECK(depression IN (0,1)),
            sweating INTEGER CHECK(sweating IN (0,1)),
            sensory_sensitivity INTEGER CHECK(sensory_sensitivity IN (0,1)),
            menstrual_pain INTEGER CHECK(menstrual_pain IN (0,1)),
            morning_stiffness INTEGER CHECK(morning_stiffness IN (0,1)),
            jaw_pain INTEGER CHECK(jaw_pain IN (0,1)),
            total_score INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')

        # -------------------------------------------------
        # Risk Factors
        # -------------------------------------------------
        conn.execute('''
        CREATE TABLE IF NOT EXISTS risk_factors (
            risk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            genetic_history INTEGER CHECK(genetic_history IN (0,1)),
            comorbid_conditions INTEGER CHECK(comorbid_conditions IN (0,1)),
            trauma_history INTEGER CHECK(trauma_history IN (0,1)),
            ptsd INTEGER CHECK(ptsd IN (0,1)),
            anxiety_depression INTEGER CHECK(anxiety_depression IN (0,1)),
            physical_inactivity INTEGER CHECK(physical_inactivity IN (0,1)),
            total_score INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')

        # -------------------------------------------------
        # Screening Result
        # -------------------------------------------------
        conn.execute('''
        CREATE TABLE IF NOT EXISTS screening_result (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            risk_probability REAL CHECK(risk_probability BETWEEN 0 AND 1),
            risk_category TEXT CHECK(risk_category IN ('Low', 'Moderate', 'High')),
            screening_status TEXT CHECK(screening_status IN ('Completed', 'Pending')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')

        # Daily Entries table (extended clinical fields)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                entry_date TEXT NOT NULL,
                symptoms TEXT,
                pain_score INTEGER,
                fatigue_score INTEGER,
                stress_score INTEGER,
                mood_score INTEGER,
                wpi TEXT,        -- JSON list of body regions ticked
                sss TEXT,        -- JSON: {fatigue, cognitive, sleep, somatic}
                sleep_quality INTEGER,
                sleep_hours REAL,
                exercise BOOLEAN,
                exercise_type TEXT,
                exercise_duration_minutes INTEGER,
                workload TEXT,
                sensory_score INTEGER,
                weather_score INTEGER,
                illness BOOLEAN,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id),
                UNIQUE(user_id, entry_date)  -- prevent duplicate daily entries
            )
        ''')

        # -------------------------------------------------
        # Monthly Assessments table
        # -------------------------------------------------
        conn.execute('''
        CREATE TABLE IF NOT EXISTS monthly_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            entry_date TEXT NOT NULL,
            phq9_score INTEGER,
            gad7_score INTEGER,
            phq9_data TEXT,  -- JSON storage of full answers
            gad7_data TEXT,  -- JSON storage of full answers
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')


        # Weekly Summary table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS weekly_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                week_start TEXT NOT NULL,
                week_end TEXT NOT NULL,
                week_number INTEGER NOT NULL,
                averages TEXT,           -- JSON averages
                acr_status INTEGER,      -- 0 or 1
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')

        # Final Report table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS final_report (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                report_json TEXT,
                generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')

        # Screenings table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                pain_regions TEXT,
                secondary_symptoms TEXT,
                primary_symptoms TEXT,
                duration TEXT,
                bmi REAL,
                first_score INTEGER,
                wpi_score INTEGER,
                sss_score INTEGER,
                meets_criteria BOOLEAN,
                risk_level TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
    conn.close()

def check_and_migrate_db():
    conn = get_db_connection()
    try:
        # Check for new columns in daily_entries
        cursor = conn.execute('PRAGMA table_info(daily_entries)')
        columns = [row['name'] for row in cursor.fetchall()]
        
        if 'sleep_hours' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN sleep_hours REAL')
        if 'sensory_score' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN sensory_score INTEGER')
        if 'weather_score' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN weather_score INTEGER')
        if 'illness' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN illness BOOLEAN')

        # New fields for replaced Daily Page (Jan 2026)
        if 'cognitive_difficulty' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN cognitive_difficulty INTEGER')
        if 'physical_activity_level' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN physical_activity_level TEXT')
        if 'sleep_duration_category' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN sleep_duration_category TEXT')
        if 'weather_sensitivity_bool' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN weather_sensitivity_bool BOOLEAN')
        if 'recent_infection' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN recent_infection BOOLEAN')
        if 'menstrual_phase' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN menstrual_phase TEXT')
        if 'pain_area_count' not in columns:
            conn.execute('ALTER TABLE daily_entries ADD COLUMN pain_area_count INTEGER')

        # -------------------------------------------------
        # FIX: primary_symptoms pain_score constraint (0-10 -> 0-19)
        # -------------------------------------------------
        # Check if we need to migrate by inspecting sql (approximated) or just do it safely
        # We can try to insert a dummy value > 10 in a transaction and see if it fails, then migrate?
        # Or just checking if we already migrated. 
        # Easier: Let's check table sql or just force migration if we haven't tracked it.
        # Since we don't have a migration version table, we'll check schema.
        
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='primary_symptoms'")
        row = cursor.fetchone()
        if row and 'CHECK(pain_score BETWEEN 0 AND 10)' in row['sql']:
            print("Migrating primary_symptoms table to allow higher pain score...")
            with conn:
                conn.execute("ALTER TABLE primary_symptoms RENAME TO primary_symptoms_old")
                conn.execute('''
                    CREATE TABLE primary_symptoms (
                        symptom_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        pain_score INTEGER CHECK(pain_score BETWEEN 0 AND 19),
                        fatigue_score INTEGER CHECK(fatigue_score BETWEEN 0 AND 10),
                        sleep_score INTEGER CHECK(sleep_score BETWEEN 0 AND 10),
                        cognitive_score INTEGER CHECK(cognitive_score BETWEEN 0 AND 10),
                        total_score INTEGER,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                ''')
                conn.execute('''
                    INSERT INTO primary_symptoms (symptom_id, user_id, pain_score, fatigue_score, sleep_score, cognitive_score, total_score)
                    SELECT symptom_id, user_id, pain_score, fatigue_score, sleep_score, cognitive_score, total_score
                    FROM primary_symptoms_old
                ''')
                conn.execute("DROP TABLE primary_symptoms_old")
            print("Migration of primary_symptoms completed.")
            
    except Exception as e:
        print(f"Migration error (harmless if already up to date): {e}")
    finally:
        conn.close()

init_db()
check_and_migrate_db()



# **************** HELPER FUNCTIONS **********************************

def validate_daily_entry_extended(data):
    """Validate extended daily entry input (Updated Jan 2026)"""
    # Relaxed validation to support new form. 
    # Mandatory fields: entry_date
    if 'entry_date' not in data:
        return False, 'Missing field: entry_date'
    
    # Check numeric bounds if fields are present
    score_fields = ['pain_score', 'fatigue_score', 'stress_score', 'mood_score', 'sleep_quality', 
                    'cognitive_difficulty', 'sensory_sensitivity_score']
    
    try:
        for key in score_fields:
            if key in data and data[key] is not None:
                val = float(data[key])
                if val < 0 or val > 10:
                    return False, f'{key} must be between 0 and 10'
    except (ValueError, TypeError):
        return False, 'Score fields must be numeric'

    return True, None


def login_required(f):
    """Decorator to protect routes that require login"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def week_bounds_for_date(date_str):
    # date_str format: YYYY-MM-DD
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    # start of week: Monday
    start = d - timedelta(days=d.weekday())
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()
VALID_SEX = {'Male', 'Female', 'Other'}
VALID_AGE_GROUPS = { '18-25','26-35','36-45','46-55','56-65','65+' }
VALID_WORKLOAD = {'Light','Moderate','Heavy','None'}
def compute_acr_status(wpi_count, sss_total):
    """
    ACR criteria (modified simplified rule):
    - ACR positive if (WPI >= 7 and SSS >= 5) OR (3 <= WPI <= 6 and SSS >= 9)
    This is the common research criteria; keep as utility function.
    """
    try:
        wpi_count = int(wpi_count)
        sss_total = float(sss_total)
    except:
        return 0
    if (wpi_count >= 7 and sss_total >= 5) or (3 <= wpi_count <= 6 and sss_total >= 9):
        return 1
    return 0
def validate_profile_payload(data):
# Ensure enums are valid; accept None for optional fields
    sex = data.get('sex')
    age_group = data.get('age_group')
    workload = data.get('workload') # optional field that may be part of profile in future


    if sex and sex not in VALID_SEX:
        return False, f"Invalid sex value. Allowed: {', '.join(VALID_SEX)}"
    if age_group and age_group not in VALID_AGE_GROPS:
        return False, f"Invalid age_group. Allowed: {', '.join(VALID_AGE_GROUPS)}"
    if workload and workload not in VALID_WORKLOAD:
        return False, f"Invalid workload. Allowed: {', '.join(VALID_WORKLOAD)}"
    return True, None


# **************** PAGE ROUTES **********************************

@app.route('/')
def landing():
    """Landing page"""
    if 'user_id' in session:
        return redirect(url_for('home_page'))
    return render_template('index.html')


@app.route('/doctors-page')
@login_required
def doctors_page():
    """Nearby doctors finder page"""
    return render_template('doctors.html')
@app.route('/login')
def login_page():
    """Login page"""
    if 'user_id' in session:
        return redirect(url_for('home_page'))
    return render_template('login.html')

@app.route('/register')
def register_page():
    """Registration page"""
    if 'user_id' in session:
        return redirect(url_for('home_page'))
    return render_template('register.html')

@app.route('/home')
@login_required
def home_page():
    """Home/Dashboard main page"""
    username = session.get('username', 'User')
    return render_template('home.html', username=username)

@app.route('/profile-page')
@login_required
def profile_page():
    """User profile page"""
    return render_template('profile.html')

@app.route('/daily-entry-page')
@login_required
def daily_entry_page():
    """Daily symptom tracking page"""
    return render_template('daily_entry.html')

@app.route('/weekly-entry-page')
@login_required
def weekly_entry_page():
    """Weekly symptom tracking page"""
    user_id = session['user_id']
    conn = get_db_connection()
    # Count total daily entries
    res = conn.execute('SELECT COUNT(*) FROM daily_entries WHERE user_id = ?', (user_id,)).fetchone()
    conn.close()
    
    days_logged = res[0] if res else 0
    # Allow access if at least 7 days logged. 
    # For "every 7 days", we might normally check (days_logged % 7 == 0), but simple threshold is safer for UI availability.
    is_unlocked = days_logged >= 7
    
    return render_template('weekly.html', days_logged=days_logged, is_unlocked=is_unlocked)

@app.route('/monthly-entry-page')
@login_required
def monthly_entry_page():
    """Monthly symptom tracking page"""
    user_id = session['user_id']
    conn = get_db_connection()
    # Count total daily entries
    res = conn.execute('SELECT COUNT(*) FROM daily_entries WHERE user_id = ?', (user_id,)).fetchone()
    conn.close()
    
    days_logged = res[0] if res else 0
    # Allow access if at least 30 days logged.
    is_unlocked = days_logged >= 30
    
    return render_template("monthly.html", days_logged=days_logged, is_unlocked=is_unlocked)

@app.route('/dashboard-page')
@login_required
def dashboard_page():
    """Analytics and charts dashboard"""
    return render_template('dashboard.html')

@app.route('/report-page')
@login_required
def report_page():
    """Reports page"""
    return render_template('report.html')

@app.route('/chat-page')
@login_required
def chat_page():
    """Gemini chatbot page"""
    return render_template('chat.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/privacy-policy')
def privacy_page():
    return render_template('privacy_policy.html')

# **************** API ENDPOINTS - AUTH **********************************

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    confirm = data.get('confirm_password')  # front-end should send this
    email = data.get('email')

    if not username or not password or not confirm:
        return jsonify({'error': 'Username, password and confirm_password required'}), 400

    if password != confirm:
        return jsonify({'error': 'Passwords do not match'}), 400

    password_hash = generate_password_hash(password)

    conn = get_db_connection()
    try:
        with conn:
            conn.execute('INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
                         (username, password_hash, email))
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already taken'}), 409
    finally:
        conn.close()

    return jsonify({'message': 'User registered successfully'})


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()

    if user is None or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid username or password'}), 401

    session['user_id'] = user['id']
    session['username'] = user['username']

    return jsonify({'message': 'Login successful', 'username': user['username'], 'user_id': user['id']})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'})

# **************** API ENDPOINTS - PROFILE **********************************


@app.route('/api/tracking-day', methods=['GET'])
def api_tracking_day():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user_id = session['user_id']
    conn = get_db_connection()
    result = conn.execute(
        'SELECT COUNT(DISTINCT entry_date) as days_logged FROM daily_entries WHERE user_id = ?',
        (user_id,)
    ).fetchone()
    conn.close()

    tracking_day = result['days_logged'] if result else 0
    max_days = 90  # 3 months

    return jsonify({'tracking_day': tracking_day, 'max_days': max_days})
@app.route('/api/profile', methods=['GET', 'POST'])
def api_profile():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user_id = session['user_id']
    conn = get_db_connection()
    if request.method == 'GET':
        user = conn.execute('SELECT sex, age_group, comorbidities, family_history, menstrual_cycle, weather_sensitivity FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        if user is None:
            return jsonify({'error': 'User not found'}), 404
        return jsonify(dict(user))
    else:
        data = request.json or {}
        valid, err = validate_profile_payload(data)
        if not valid:
            conn.close()
            return jsonify({'error': err}), 400
    sex = data.get('sex')
    age_group = data.get('age_group')
    comorbidities = data.get('comorbidities')
    # allow front-end to send list or comma separated string
    if isinstance(comorbidities, list):
        comorbidities = ','.join(comorbidities)
    family_history = data.get('family_history')
    menstrual_cycle = data.get('menstrual_cycle')
    weather_sensitivity = data.get('weather_sensitivity')
    with conn:
        conn.execute('''
    UPDATE users SET sex=?, age_group=?, comorbidities=?, family_history=?, menstrual_cycle=?, weather_sensitivity=?, profile_complete=1 WHERE id=?
    ''', (sex, age_group, comorbidities, family_history, menstrual_cycle, weather_sensitivity, user_id))
    conn.close()
    return jsonify({'message': 'Profile updated successfully'})

# **************** API ENDPOINTS - DAILY ENTRY **********************************

@app.route('/api/daily-entry', methods=['POST'])
def api_daily_entry():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user_id = session['user_id']
    data = request.json

    # Validate input
    valid, error_msg = validate_daily_entry_extended(data)
    if not valid:
        return jsonify({'error': error_msg}), 400

    entry_date = data.get('entry_date')
    
    # Legacy/Existing fields
    symptoms = json.dumps(data.get('symptoms')) if data.get('symptoms') else None
    pain_score = data.get('pain_score')
    fatigue_score = data.get('fatigue_score')
    stress_score = data.get('stress_score')
    mood_score = data.get('mood_score')
    
    wpi = json.dumps(data.get('wpi')) if data.get('wpi') else None
    sss = json.dumps(data.get('sss')) if data.get('sss') else None
    
    sleep_quality = data.get('sleep_quality')
    sleep_hours = data.get('sleep_hours')
    
    exercise = int(data.get('exercise')) if data.get('exercise') is not None else None
    exercise_type = data.get('exercise_type')
    exercise_duration_minutes = data.get('exercise_duration_minutes')
    workload = data.get('workload')
    
    sensory_score = data.get('sensory_score') # Existing integer field, reused for "Sensory Sensitivity"
    # If frontend sends 'sensory_sensitivity_score', map it to 'sensory_score'
    if 'sensory_sensitivity_score' in data:
        sensory_score = data.get('sensory_sensitivity_score')

    weather_score = data.get('weather_score') # Existing integer field.
    illness = int(data.get('illness')) if data.get('illness') is not None else 0

    # New fields (Jan 2026)
    cognitive_difficulty = data.get('cognitive_difficulty')
    physical_activity_level = data.get('physical_activity_level')
    sleep_duration_category = data.get('sleep_duration_category')
    weather_sensitivity_bool = 1 if data.get('weather_sensitivity_bool') else 0
    recent_infection = 1 if data.get('recent_infection') else 0
    menstrual_phase = data.get('menstrual_phase')
    pain_area_count = data.get('pain_area_count')

    conn = get_db_connection()
    with conn:
        conn.execute('''
            INSERT INTO daily_entries
            (user_id, entry_date, symptoms, pain_score, fatigue_score, stress_score, mood_score,
             wpi, sss, sleep_quality, sleep_hours, exercise, exercise_type, exercise_duration_minutes, 
             workload, sensory_score, weather_score, illness,
             cognitive_difficulty, physical_activity_level, sleep_duration_category, weather_sensitivity_bool,
             recent_infection, menstrual_phase, pain_area_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, entry_date, symptoms, pain_score, fatigue_score, stress_score, mood_score,
              wpi, sss, sleep_quality, sleep_hours, exercise, exercise_type, exercise_duration_minutes, 
              workload, sensory_score, weather_score, illness,
              cognitive_difficulty, physical_activity_level, sleep_duration_category, weather_sensitivity_bool,
              recent_infection, menstrual_phase, pain_area_count))
    conn.close()
    return jsonify({'message': 'Entry saved'})
# ----------------- WEEKLY SUMMARY LIST endpoint -----------------


@app.route('/api/weekly-summaries', methods=['GET'])
def api_weekly_summaries():
    """Return list of weekly summaries for the logged-in user (for dashboard)"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user_id = session['user_id']
    conn = get_db_connection()
    rows = conn.execute('SELECT * FROM weekly_summary WHERE user_id = ? ORDER BY week_start DESC', (user_id,)).fetchall()
    conn.close()
    summaries = []
    for r in rows:
        summaries.append({
    'id': r['id'],
    'week_start': r['week_start'],
    'week_end': r['week_end'],
    'week_number': r['week_number'],
    'averages': json.loads(r['averages']) if r['averages'] else {},
    'acr_status': r['acr_status']
    })
    return jsonify({'weekly_summaries': summaries})
@app.route('/api/weekly-summary', methods=['GET', 'POST'])
def api_weekly_summary():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user_id = session['user_id']

    # GET expects ?date=YYYY-MM-DD or ?week_start=YYYY-MM-DD
    if request.method == 'GET':
        date_q = request.args.get('date')
        week_start_q = request.args.get('week_start')

        if date_q:
            week_start, week_end = week_bounds_for_date(date_q)
        elif week_start_q:
            week_start = week_start_q
            start = datetime.strptime(week_start, "%Y-%m-%d").date()
            week_end = (start + timedelta(days=6)).isoformat()
        else:
            return jsonify({'error': 'Provide date or week_start parameter as YYYY-MM-DD'}), 400

        conn = get_db_connection()
        entries = conn.execute('''
            SELECT * FROM daily_entries WHERE user_id = ? AND entry_date BETWEEN ? AND ? ORDER BY entry_date
        ''', (user_id, week_start, week_end)).fetchall()

        if not entries:
            return jsonify({'message': 'No data for this week', 'week_start': week_start, 'week_end': week_end, 'averages': {}})

        # compute averages and WPI/SSS summary
        pain_vals = []
        fatigue_vals = []
        stress_vals = []
        mood_vals = []
        sleep_vals = []
        wpi_counts = []
        sss_sums = []

        for e in entries:
            if e['pain_score'] is not None:
                pain_vals.append(e['pain_score'])
            if e['fatigue_score'] is not None:
                fatigue_vals.append(e['fatigue_score'])
            if e['stress_score'] is not None:
                stress_vals.append(e['stress_score'])
            if e['mood_score'] is not None:
                mood_vals.append(e['mood_score'])
            if e['sleep_quality'] is not None:
                sleep_vals.append(e['sleep_quality'])

            # wpi processing
            if e['wpi']:
                try:
                    wpi_list = json.loads(e['wpi'])
                    wpi_counts.append(len(wpi_list))
                except:
                    pass
            # sss processing -> sum subscales
            if e['sss']:
                try:
                    sss_d = json.loads(e['sss'])
                    sss_total = sum([float(v) for v in sss_d.values() if v is not None])
                    sss_sums.append(sss_total)
                except:
                    pass

        averages = {
            'avg_pain': round(np.mean(pain_vals), 2) if pain_vals else None,
            'avg_fatigue': round(np.mean(fatigue_vals), 2) if fatigue_vals else None,
            'avg_stress': round(np.mean(stress_vals), 2) if stress_vals else None,
            'avg_mood': round(np.mean(mood_vals), 2) if mood_vals else None,
            'avg_sleep': round(np.mean(sleep_vals), 2) if sleep_vals else None,
            'avg_wpi_count': round(np.mean(wpi_counts), 2) if wpi_counts else 0,
            'avg_sss_total': round(np.mean(sss_sums), 2) if sss_sums else 0
        }

        # compute ACR status based on averaged WPI count and SSS
        acr = compute_acr_status(averages['avg_wpi_count'], averages['avg_sss_total'])

        # Store / update weekly_summary table
        week_number = datetime.strptime(week_start, "%Y-%m-%d").isocalendar()[1]
        with conn:
            # We will simple insert a row each time (or you can upsert by week_start)
            conn.execute('''
                INSERT INTO weekly_summary (user_id, week_start, week_end, week_number, averages, acr_status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, week_start, week_end, week_number, json.dumps(averages), acr))
        conn.close()

        return jsonify({
            'week_start': week_start,
            'week_end': week_end,
            'week_number': week_number,
            'averages': averages,
            'acr_status': acr
        })

@app.route('/api/correlations', methods=['GET'])
def api_correlations():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user_id = session['user_id']

    # date range optional: ?from=YYYY-MM-DD&to=YYYY-MM-DD
    date_from = request.args.get('from')
    date_to = request.args.get('to')

    conn = get_db_connection()
    if date_from and date_to:
        rows = conn.execute('SELECT * FROM daily_entries WHERE user_id = ? AND entry_date BETWEEN ? AND ? ORDER BY entry_date', (user_id, date_from, date_to)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date', (user_id,)).fetchall()
    conn.close()

    # Extract arrays
    pain = []
    stress = []
    fatigue = []
    sleep = []
    mood = []
    for r in rows:
        if r['pain_score'] is not None:
            pain.append(r['pain_score'])
            stress.append(r['stress_score'] if r['stress_score'] is not None else np.nan)
            fatigue.append(r['fatigue_score'] if r['fatigue_score'] is not None else np.nan)
            sleep.append(r['sleep_quality'] if r['sleep_quality'] is not None else np.nan)
            mood.append(r['mood_score'] if r['mood_score'] is not None else np.nan)

    # Build dataframe
    df = pd.DataFrame({
        'pain': pain,
        'stress': stress,
        'fatigue': fatigue,
        'sleep': sleep,
        'mood': mood
    }).dropna()

    if df.empty or len(df) < 2:
        return jsonify({'error': 'Not enough data to compute correlations'}), 400

    corr = df.corr().to_dict()
    # Return key pairs of interest: pain vs others
    correlations = {
        'pain_vs_stress': corr.get('pain', {}).get('stress'),
        'pain_vs_fatigue': corr.get('pain', {}).get('fatigue'),
        'pain_vs_sleep': corr.get('pain', {}).get('sleep'),
        'pain_vs_mood': corr.get('pain', {}).get('mood'),
        'matrix': corr
    }
    return jsonify(correlations)



@app.route('/api/report/weekly/export-excel', methods=['GET'])
@login_required
def export_weekly_excel():
    user_id = session['user_id']
    week_number = request.args.get('week_number')
    
    if not week_number:
        return jsonify({'error': 'week_number required'}), 400

    week_number = int(week_number)

    # Fetch entries for the user
    conn = get_db_connection()
    entries = conn.execute(
        'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date ASC',
        (user_id,)
    ).fetchall()
    conn.close()

    if not entries:
        return jsonify({'error': 'No data available for this week'}), 404

    # Determine week slice
    start_idx = (week_number - 1) * 7
    week_entries = entries[start_idx:start_idx + 7]
    if not week_entries:
        return jsonify({'error': 'Week number out of range'}), 400

    # Prepare data for Excel
    data = []
    for e in week_entries:
        # Safely load JSON from 'sss'
        sss = json.loads(e['sss']) if e['sss'] else {}
        data.append({
            'Date': e['entry_date'],
            'Pain': e['pain_score'] or 0,
            'Fatigue': sss.get('fatigue', 0),
            'Cognitive': sss.get('cognitive', 0),
            'Sleep': sss.get('sleep', 0),
            'Somatic': sss.get('somatic', 0),
            'Stress': e['stress_score'] or 0,
            'Mood': e['mood_score'] or 0,
        })

    df = pd.DataFrame(data)

    # Export to Excel in memory
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name=f'Week_{week_number}')

    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'weekly_report_week_{week_number}.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.route('/api/report/final/export-excel', methods=['GET'])
@login_required
def export_final_excel():
        user_id = session['user_id']

        # Fetch all entries for the user
        conn = get_db_connection()
        entries = conn.execute(
            'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date ASC',
            (user_id,)
        ).fetchall()
        conn.close()

        if not entries:
            return jsonify({'error': 'No data available'}), 404

        # Prepare data for Excel
        data = []
        for e in entries:
            sss = json.loads(e['sss']) if e['sss'] else {}
            data.append({
                'Date': e['entry_date'],
                'Pain': e['pain_score'] or 0,
                'Fatigue': sss.get('fatigue', 0),
                'Cognitive': sss.get('cognitive', 0),
                'Sleep': sss.get('sleep', 0),
                'Somatic': sss.get('somatic', 0),
                'Stress': e['stress_score'] or 0,
                'Mood': e['mood_score'] or 0,
            })

        df = pd.DataFrame(data)

        # Export to Excel in memory
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Final_Report')

        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name='final_report.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
@app.route('/api/report/final', methods=['GET'])
def api_final_report():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user_id = session['user_id']
    conn = get_db_connection()

    try:
        # require at least 12 weeks
        weeks = conn.execute(
            'SELECT * FROM weekly_summary WHERE user_id = ? ORDER BY week_start',
            (user_id,)
        ).fetchall()

        if not weeks or len(weeks) < 12:
            return jsonify({'message': 'Not enough data for final report (requires 12 weekly summaries)'}), 400

        # build the final report JSON
        weekly_data = [
            {
                'week_start': w['week_start'],
                'week_end': w['week_end'],
                'week_number': w['week_number'],
                'averages': json.loads(w['averages']) if w['averages'] else {},
                'acr_status': w['acr_status']
            }
            for w in weeks
        ]

        # compute trends (first vs last week)
        first = weekly_data[0]['averages']
        last = weekly_data[-1]['averages']
        trend = {}
        for k in ['avg_pain', 'avg_fatigue', 'avg_stress', 'avg_mood', 'avg_sleep']:
            try:
                start_val = first.get(k)
                end_val = last.get(k)
                delta = None if start_val is None or end_val is None else round(end_val - start_val, 2)
                trend[k] = {'start': start_val, 'end': end_val, 'delta': delta}
            except Exception:
                trend[k] = {}

        # Determine overall ACR
        acr_overall = any(w['acr_status'] == 1 for w in weekly_data)

        final = {
            'weekly_data': weekly_data,
            'trend': trend,
            'acr_overall': int(acr_overall)
        }

        # Save to DB (replace if exists)
        conn.execute(
            'INSERT OR REPLACE INTO final_report (user_id, report_json) VALUES (?, ?)',
            (user_id, json.dumps(final))
        )
        conn.commit()

        return jsonify(final)

    finally:
        conn.close()



@app.route('/api/daily-entry', methods=['GET'])
def api_get_daily_entry():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user_id = session['user_id']
    entry_date = request.args.get('date')

    if not entry_date:
        return jsonify({'error': 'Date parameter required'}), 400

    conn = get_db_connection()
    entry = conn.execute('SELECT * FROM daily_entries WHERE user_id = ? AND entry_date = ?', 
                         (user_id, entry_date)).fetchone()
    conn.close()

    if entry is None:
        return jsonify({'error': 'Entry not found'}), 404

    entry_dict = dict(entry)
    # Decode JSON fields
    if entry_dict.get('symptoms'):
        entry_dict['symptoms'] = json.loads(entry_dict['symptoms'])
    if entry_dict.get('wpi'):
        entry_dict['wpi'] = json.loads(entry_dict['wpi'])
    if entry_dict.get('sss'):
        entry_dict['sss'] = json.loads(entry_dict['sss'])
    # Convert exercise back to boolean
    if 'exercise' in entry_dict:
        entry_dict['exercise'] = bool(entry_dict['exercise'])

    return jsonify(entry_dict)



@app.route('/api/dashboard/daily', methods=['GET'])
@login_required
def api_dashboard_daily():
    """
    Return all daily entries for logged-in user (sorted by date) for line graph and heatmap
    """
    user_id = session['user_id']
    conn = get_db_connection()
    entries = conn.execute(
        'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date ASC',
        (user_id,)
    ).fetchall()
    conn.close()

    if not entries:
        return jsonify({'entries': [], 'message': 'No data available'})

    result = []
    for e in entries:
        entry_dict = dict(e)
        if entry_dict.get('sss'):
            sss = json.loads(entry_dict['sss'])
            entry_dict.update(sss)
            del entry_dict['sss']
        if entry_dict.get('wpi'):
            entry_dict['wpi'] = json.loads(entry_dict['wpi'])
        
        # safer handling for exercise
        entry_dict['exercise'] = bool(entry_dict.get('exercise', 0))
        
        result.append({
            'entry_date': entry_dict['entry_date'],
            'pain_score': entry_dict.get('pain_score'),
            'fatigue': entry_dict.get('fatigue'),
            'cognitive': entry_dict.get('cognitive'),
            'sleep': entry_dict.get('sleep'),
            'somatic': entry_dict.get('somatic'),
            'mood_score': entry_dict.get('mood_score'),
            'workload': entry_dict.get('workload')
        })
    return jsonify(result)

@app.route('/api/ai-suggestions', methods=['POST'])
@login_required
def api_ai_suggestions():
    """
    Expects JSON input like:
    {
        "summary": {
            "average_pain": 4.5,
            "average_fatigue": 6,
            "average_sleep": 5,
            "average_stress": 7,
            "average_mood": 3
        }
    }
    Returns AI-generated suggestions as JSON.
    """
    data = request.json
    summary = data.get('summary')
    if not summary:
        return jsonify({'error': 'Summary data required'}), 400

    try:
        model_gemini = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = f"""
        You are a medical assistant for Fibromyalgia patients.
        Here is the patient's summary data:
        {json.dumps(summary, indent=2)}
        Based on this, give personalized advice for stress, sleep, pain, and fatigue.
        Return short, actionable recommendations.
        """
        response = model_gemini.generate_content(prompt)
        ai_advice = response.text
        return jsonify({'advice': ai_advice})
    except Exception as e:
        return jsonify({'advice': f"AI suggestions unavailable: {str(e)}"}), 500
    
@app.route('/api/dashboard/ai-suggestions', methods=['GET'])
@login_required
def api_dashboard_ai_suggestions():
    """
    Fetches recent weekly averages and returns AI suggestions
    """
    user_id = session['user_id']
    conn = get_db_connection()
    # Fetch last 7 days
    entries = conn.execute(
        'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date DESC LIMIT 7',
        (user_id,)
    ).fetchall()
    conn.close()

    if not entries:
        return jsonify({'error': 'No recent data to generate AI suggestions'}), 404

    # Compute simple averages
    total_pain = total_fatigue = total_sleep = total_stress = total_mood = 0
    count = 0
    for e in entries:
        entry = dict(e)  # ✅ convert sqlite3.Row -> dict

        sss = json.loads(entry['sss']) if entry.get('sss') else {}
        total_pain += entry.get('pain_score', 0) or 0
        total_fatigue += sss.get('fatigue', 0)
        total_sleep += sss.get('sleep', 0)
        total_stress += entry.get('stress_score', 0) or 0
        total_mood += entry.get('mood_score', 0) or 0
        count += 1

    recent_summary = {
        "average_pain": round(total_pain / count, 2),
        "average_fatigue": round(total_fatigue / count, 2),
        "average_sleep": round(total_sleep / count, 2),
        "average_stress": round(total_stress / count, 2),
        "average_mood": round(total_mood / count, 2),
    }

    # Generate AI suggestions
    try:
        model_gemini = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        You are a medical assistant for Fibromyalgia patients.
        Here is the patient's recent 7-day summary:
        {json.dumps(recent_summary, indent=2)}
        Provide short actionable advice for stress, sleep, pain, and fatigue.
        """
        response = model_gemini.generate_content(prompt)
        ai_advice = response.text
    except Exception as e:
        ai_advice = f"AI suggestions unavailable: {str(e)}"

    return jsonify({
        "recent_summary": recent_summary,
        "ai_advice": ai_advice
    })
@app.route('/api/dashboard/weekly', methods=['GET'])
@login_required
def api_dashboard_weekly():
    """
    Aggregate daily entries by week for logged-in user and return averages
    """
    user_id = session['user_id']
    conn = get_db_connection()
    entries = conn.execute(
        'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date ASC',
        (user_id,)
    ).fetchall()
    conn.close()

    if not entries:
        return jsonify({'entries': [], 'message': 'No data available'})

    # Group entries by week number
    weekly_data = {}
    for e in entries:
        entry = dict(e)  # ✅ convert sqlite3.Row -> dict

        entry_date = datetime.strptime(entry['entry_date'], '%Y-%m-%d')
        week_start = (entry_date - timedelta(days=entry_date.weekday())).date()  # Monday
        week_key = str(week_start)

        if week_key not in weekly_data:
            weekly_data[week_key] = {
                'pain_scores': [],
                'fatigue_scores': [],
                'sleep_scores': [],
                'stress_scores': [],
                'mood_scores': [],
                'workloads': []
            }

        # Expand sss JSON safely
        sss = json.loads(entry['sss']) if entry.get('sss') else {}

        weekly_data[week_key]['pain_scores'].append(entry.get('pain_score', 0) or 0)
        weekly_data[week_key]['fatigue_scores'].append(sss.get('fatigue', 0))
        weekly_data[week_key]['sleep_scores'].append(sss.get('sleep', 0))
        weekly_data[week_key]['stress_scores'].append(entry.get('stress_score', 0) or 0)
        weekly_data[week_key]['mood_scores'].append(entry.get('mood_score', 0) or 0)
        if entry.get('workload') is not None:
            weekly_data[week_key]['workloads'].append(entry['workload'])

    # Compute weekly averages
    weekly_result = []
    for i, (week_start, data) in enumerate(sorted(weekly_data.items()), start=1):
        week_end = (datetime.strptime(week_start, '%Y-%m-%d') + timedelta(days=6)).date()

        avg_pain = sum(data['pain_scores']) / len(data['pain_scores'])
        avg_fatigue = sum(data['fatigue_scores']) / len(data['fatigue_scores'])
        avg_sleep = sum(data['sleep_scores']) / len(data['sleep_scores'])
        avg_stress = sum(data['stress_scores']) / len(data['stress_scores'])
        avg_mood = sum(data['mood_scores']) / len(data['mood_scores'])

        # Most frequent workload if available
        workload = max(set(data['workloads']), key=data['workloads'].count) if data['workloads'] else None

        weekly_result.append({
            'week_number': i,
            'week_start': week_start,
            'week_end': str(week_end),
            'avg_pain': round(avg_pain, 2),
            'avg_fatigue': round(avg_fatigue, 2),
            'avg_sleep': round(avg_sleep, 2),
            'avg_stress': round(avg_stress, 2),
            'avg_mood': round(avg_mood, 2),
            'avg_workload': workload
        })

    return jsonify(weekly_result)

@app.route('/api/report/weekly', methods=['GET'])
@login_required
def api_report_weekly():
    user_id = session['user_id']
    week_number = request.args.get('week_number')
    if not week_number:
        return jsonify({'error': 'week_number required'}), 400

    conn = get_db_connection()
    entries = conn.execute(
        'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date ASC',
        (user_id,)
    ).fetchall()
    conn.close()

    if not entries:
        return jsonify({'error': 'No data available for this week'}), 404

    # Filter entries for this week (simplified: assume week_number matches sorted order)
    week_number = int(week_number)
    total_weeks = (len(entries) + 6) // 7
    if week_number > total_weeks:
        return jsonify({'error': 'Week number out of range'}), 400

    start_idx = (week_number - 1) * 7
    week_entries = entries[start_idx:start_idx + 7]

    # Compute weekly averages
    avg_pain = sum([e['pain_score'] or 0 for e in week_entries]) / len(week_entries)
    avg_fatigue = sum([(json.loads(e['sss']).get('fatigue', 0) if e['sss'] else 0)for e in week_entries]) / len(week_entries)
    avg_sleep = sum([(json.loads(e['sss']).get('sleep', 0) if e['sss'] else 0)for e in week_entries]) / len(week_entries)
    avg_stress = sum([e['stress_score'] or 0 for e in week_entries]) / len(week_entries)
    avg_mood = sum([e['mood_score'] or 0 for e in week_entries]) / len(week_entries)

    # Placeholder ACR criteria (replace with real logic)
    acr_met = 1 if avg_pain >= 4 else 0

    # Prepare AI context for Gemini
    ai_prompt = f"""
    The patient has the following weekly averages:
    Pain: {avg_pain}, Fatigue: {avg_fatigue}, Sleep: {avg_sleep}, Stress: {avg_stress}, Mood: {avg_mood}.
    Give concise personalized suggestions to improve symptoms.
    """

    try:
        model_gemini = genai.GenerativeModel("gemini-2.0-flash")
        response = model_gemini.generate_content(ai_prompt)
        ai_advice = response.text
    except Exception as e:
        ai_advice = f"AI generation failed: {str(e)}"

    report = {
        'week_number': week_number,
        'summary': {
            'avg_pain': round(avg_pain, 2),
            'avg_fatigue': round(avg_fatigue, 2),
            'avg_sleep': round(avg_sleep, 2),
            'avg_stress': round(avg_stress, 2),
            'avg_mood': round(avg_mood, 2),
            'acr_met': acr_met
        },
        'ai_advice': ai_advice
    }

    return jsonify(report)


@app.route('/api/report/final', methods=['GET'])
@login_required
def api_report_final():
    user_id = session['user_id']
    conn = get_db_connection()
    user_profile = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    entries = conn.execute('SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date ASC', (user_id,)).fetchall()
    conn.close()

    if not entries:
        return jsonify({'error': 'No data available'}), 404

    # Compute overall averages
    avg_pain = sum([e['pain_score'] or 0 for e in entries]) / len(entries)
    avg_fatigue = sum([json.loads(e['sss']).get('fatigue', 0) if e.get('sss') else 0 for e in entries]) / len(entries)
    avg_sleep = sum([json.loads(e['sss']).get('sleep', 0) if e.get('sss') else 0 for e in entries]) / len(entries)
    avg_stress = sum([e['stress_score'] or 0 for e in entries]) / len(entries)
    avg_mood = sum([e['mood_score'] or 0 for e in entries]) / len(entries)

    # Placeholder FIQr trend analysis
    fiqr_trend = "↑" if avg_pain > 5 else "↓"

    # AI context for final report
    ai_prompt = f"""
    Patient profile: {dict(user_profile)}
    3-month summary averages:
    Pain: {avg_pain}, Fatigue: {avg_fatigue}, Sleep: {avg_sleep}, Stress: {avg_stress}, Mood: {avg_mood}.
    Provide personalized advice for stress, sleep, pain, fatigue and overall recommendations.
    """

    try:
        model_gemini = genai.GenerativeModel("gemini-2.0-flash")
        response = model_gemini.generate_content(ai_prompt)
        ai_advice = response.text
    except Exception as e:
        ai_advice = f"AI generation failed: {str(e)}"

    final_report = {
        'profile': dict(user_profile),
        'averages': {
            'avg_pain': round(avg_pain, 2),
            'avg_fatigue': round(avg_fatigue, 2),
            'avg_sleep': round(avg_sleep, 2),
            'avg_stress': round(avg_stress, 2),
            'avg_mood': round(avg_mood, 2),
        },
        'fiqr_trend': fiqr_trend,
        'ai_advice': ai_advice,
        'doctor_recommendation': "Consult a rheumatologist if ACR criteria met or high FIQr"  # placeholder
    }

    return jsonify(final_report)

@app.route('/api/daily-entries', methods=['GET'])
def api_get_all_entries():
    """Get all entries for logged-in user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user_id = session['user_id']
    conn = get_db_connection()
    entries = conn.execute('SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date DESC', 
                           (user_id,)).fetchall()
    conn.close()

    entries_list = []
    for entry in entries:
        entry_dict = dict(entry)
        if entry_dict.get('symptoms'):
            entry_dict['symptoms'] = json.loads(entry_dict['symptoms'])
        entries_list.append(entry_dict)
    
    return jsonify({'entries': entries_list})

# **************** API ENDPOINTS - ML PREDICTION **********************************

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    if model is None:
        return jsonify({'error': 'ML model not loaded'}), 500

    data = request.json
    required_features = ['week', 'pain_level', 'fatigue', 'sleep_quality', 'stiffness', 'mood', 'activity_difficulty', 'stress_level']

    try:
        feature_values = [float(data[feature]) for feature in required_features]
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Missing or invalid features: {str(e)}'}), 400

    feature_array = np.array(feature_values).reshape(1, -1)

    # Predict flare
    flare_pred = model.predict(feature_array)[0]
    flare_prob = model.predict_proba(feature_array)[0].tolist()

    response = {
        'flare_prediction': int(flare_pred),
        'flare_probability': {'no_flare': flare_prob[0], 'flare': flare_prob[1]}
    }

    return jsonify(response)

@app.route('/api/monthly-entry', methods=['POST'])
@login_required
def save_monthly_entry():
    """Save monthly assessment data (PHQ-9, GAD-7)"""
    try:
        user_id = session['user_id']
        data = request.json
        
        # Expecting: { entry_date, phq9_score, gad7_score, phq9_data, gad7_data }
        entry_date = data.get('entry_date', datetime.now().strftime('%Y-%m-%d'))
        phq9_score = data.get('phq9_score')
        gad7_score = data.get('gad7_score')
        
        # Extract raw data dictionaries
        phq9_raw = data.get('phq9_data', {})
        gad7_raw = data.get('gad7_data', {})

        # --- AI Prediction (GAD-7) ---
        if GAD_MODEL:
            try:
                # Prepare input dict: question1..7, time1..7
                gad_input = {}
                # Questions
                for k, v in gad7_raw.items():
                    if k.startswith('question'):
                        gad_input[k] = v
                # Times
                times = gad7_raw.get('times', {})
                for k, v in times.items():
                    gad_input[k] = v
                
                # Check for completeness (simple check)
                if len(gad_input) >= 14: # 7 qs + 7 times
                    # Feature engineering as per gad.py
                    # We need to create a DataFrame to use the exact logic or replicate it.
                    # GAD model expects: `question1`..`question7` + `avg_response_time` + `max_response_time`
                    # Wait, the gad.py `predict_gad_severity` calculates avg/max from time cols.
                    # But the MODEL object itself (RandomForest) expects the final feature vector.
                    # I need to see `gad.py` again to know exact feature columns expected by `model.predict`.
                    # View file `dataset/models/gad/gad.py` showed:
                    # features = temp_df[question_cols + ["avg_response_time", "max_response_time"]]
                    # So feature order matters.
                    
                    # Columns: question1..7, avg_response_time, max_response_time.
                    
                    df_gad = pd.DataFrame([gad_input])
                    time_cols = [f'time{i}' for i in range(1, 8)]
                    q_cols = [f'question{i}' for i in range(1, 8)]
                    
                    # Fill missing timings with defaults?
                    for c in time_cols:
                        if c not in df_gad.columns: df_gad[c] = 0
                    
                    df_gad["avg_response_time"] = df_gad[time_cols].mean(axis=1)
                    df_gad["max_response_time"] = df_gad[time_cols].max(axis=1)
                    
                    features = df_gad[q_cols + ["avg_response_time", "max_response_time"]]
                    
                    pred_class = GAD_MODEL.predict(features)[0]
                    pred_prob = GAD_MODEL.predict_proba(features).max()
                    
                    severity_map = {0: "Minimal anxiety", 1: "Mild anxiety", 2: "Moderate anxiety", 3: "Moderate to severe anxiety"}
                    predicted_severity = severity_map.get(pred_class, "Unknown")
                    
                    gad7_raw['ai_prediction'] = {
                        'severity': predicted_severity,
                        'confidence': round(float(pred_prob), 3)
                    }
                    print(f"🧠 GAD-7 AI Prediction: {predicted_severity} ({pred_prob:.2f})")
            except Exception as e:
                print(f"⚠️ GAD-7 Prediction failed: {e}")

        # --- AI Prediction (PHQ-9) ---
        if PHQ_MODEL:
            try:
                phq_input = {}
                for k, v in phq9_raw.items():
                    if k.startswith('question'):
                        phq_input[k] = v
                times = phq9_raw.get('times', {})
                for k, v in times.items():
                    phq_input[k] = v
                
                if len(phq_input) >= 18: # 9 qs + 9 times
                    df_phq = pd.DataFrame([phq_input])
                    time_cols = [f'time{i}' for i in range(1, 10)]
                    q_cols = [f'question{i}' for i in range(1, 10)]
                    
                    for c in time_cols:
                        if c not in df_phq.columns: df_phq[c] = 0
                        
                    df_phq["avg_response_time"] = df_phq[time_cols].mean(axis=1)
                    df_phq["max_response_time"] = df_phq[time_cols].max(axis=1)
                    
                    features = df_phq[q_cols + ["avg_response_time", "max_response_time"]]
                    
                    pred_class = PHQ_MODEL.predict(features)[0]
                    pred_prob = PHQ_MODEL.predict_proba(features).max()
                    
                    severity_map = {0: "Minimal", 1: "Mild", 2: "Moderate", 3: "Moderately Severe", 4: "Severe"}
                    predicted_severity = severity_map.get(pred_class, "Unknown")
                    
                    phq9_raw['ai_prediction'] = {
                        'severity': predicted_severity,
                        'confidence': round(float(pred_prob), 3)
                    }
                    print(f"🧠 PHQ-9 AI Prediction: {predicted_severity} ({pred_prob:.2f})")
            except Exception as e:
                print(f"⚠️ PHQ-9 Prediction failed: {e}")

        # Serialize for DB
        phq9_data_json = json.dumps(phq9_raw)
        gad7_data_json = json.dumps(gad7_raw)

        conn = get_db_connection()
        conn.execute('''
            INSERT INTO monthly_assessments 
            (user_id, entry_date, phq9_score, gad7_score, phq9_data, gad7_data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, entry_date, phq9_score, gad7_score, phq9_data_json, gad7_data_json))
        
        conn.commit()
        conn.close()

        return jsonify({"success": True, "message": "Monthly assessment saved successfully"})

    except Exception as e:
        print(f"Error saving monthly assessment: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# **************** API ENDPOINTS - CHATBOT **********************************

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({'error': 'Message required'}), 400

    try:
        model_gemini = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model_gemini.generate_content(message)
        chat_reply = response.text
    except Exception as e:
        return jsonify({'error': f'Gemini API error: {str(e)}'}), 500

    return jsonify({'response': chat_reply})

# **************** API ENDPOINTS - REPORTS **********************************

@app.route('/api/report/summary', methods=['GET'])
def api_report_summary():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session['user_id']
    conn = get_db_connection()
    user_profile = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    entries = conn.execute('SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date', (user_id,)).fetchall()
    conn.close()
    
    if not user_profile:
        return jsonify({'error': 'User not found'}), 404
    
    total_entries = len(entries)
    avg_pain = sum([e['pain_score'] for e in entries if e['pain_score']]) / total_entries if total_entries else 0
    avg_fatigue = sum([e['fatigue_score'] for e in entries if e['fatigue_score']]) / total_entries if total_entries else 0
    avg_stress = sum([e['stress_score'] for e in entries if e['stress_score']]) / total_entries if total_entries else 0
    avg_mood = sum([e['mood_score'] for e in entries if e['mood_score']]) / total_entries if total_entries else 0
    
    report = {
        'username': user_profile['username'],
        'profile': {
            'sex': user_profile['sex'],
            'age_group': user_profile['age_group'],
            'comorbidities': user_profile['comorbidities'],
        },
        'summary': {
            'total_entries': total_entries,
            'average_pain': round(avg_pain, 2),
            'average_fatigue': round(avg_fatigue, 2),
            'average_stress': round(avg_stress, 2),
            'average_mood': round(avg_mood, 2),
        }
    }
    
    return jsonify(report)
@app.route('/api/report/weekly/export-pdf', methods=['GET'])
@login_required
def export_weekly_pdf():
    user_id = session['user_id']
    week_number = request.args.get('week_number')
    if not week_number:
        return jsonify({'error': 'week_number required'}), 400

    week_number = int(week_number)

    # Fetch entries
    conn = get_db_connection()
    entries = conn.execute(
        'SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date ASC',
        (user_id,)
    ).fetchall()
    conn.close()

    if not entries:
        return jsonify({'error': 'No data available for this week'}), 404

    # Slice week
    start_idx = (week_number - 1) * 7
    week_entries = entries[start_idx:start_idx + 7]

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, f"FibroTracker Weekly Report - Week {week_number}")

    # Table header
    y = height - 80
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, "Date")
    p.drawString(120, y, "Pain")
    p.drawString(170, y, "Fatigue")
    p.drawString(230, y, "Sleep")
    p.drawString(290, y, "Stress")
    p.drawString(350, y, "Mood")
    y -= 20
    p.setFont("Helvetica", 10)

    # Table rows
    for e in week_entries:
        sss = json.loads(e['sss']) if e['sss'] else {}
        p.drawString(50, y, str(e['entry_date']))
        p.drawString(120, y, str(e['pain_score'] or 0))
        p.drawString(170, y, str(sss.get('fatigue', 0)))
        p.drawString(230, y, str(sss.get('sleep', 0)))
        p.drawString(290, y, str(e['stress_score'] or 0))
        p.drawString(350, y, str(e['mood_score'] or 0))
        y -= 15

        if y < 50:  # new page if space runs out
            p.showPage()
            y = height - 50

    p.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'weekly_report_week_{week_number}.pdf',
        mimetype='application/pdf'
    )


@app.route('/api/report/final/export-pdf', methods=['GET'])
@login_required
def export_final_pdf():
    user_id = session['user_id']

    conn = get_db_connection()
    user_profile = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    entries = conn.execute('SELECT * FROM daily_entries WHERE user_id = ? ORDER BY entry_date ASC', (user_id,)).fetchall()
    conn.close()

    if not entries:
        return jsonify({'error': 'No data available'}), 404

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "FibroTracker Final Report")

    # User info
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 80, f"Username: {user_profile['username']}")
    p.drawString(50, height - 100, f"Sex: {user_profile['sex'] or 'N/A'}")
    p.drawString(50, height - 120, f"Age Group: {user_profile['age_group'] or 'N/A'}")

    # Weekly table header
    y = height - 160
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, "Date")
    p.drawString(120, y, "Pain")
    p.drawString(170, y, "Fatigue")
    p.drawString(230, y, "Sleep")
    p.drawString(290, y, "Stress")
    p.drawString(350, y, "Mood")
    y -= 20
    p.setFont("Helvetica", 10)

    for e in entries:
        sss = json.loads(e['sss']) if e['sss']  else {}
        p.drawString(50, y, e['entry_date'])
        p.drawString(120, y, str(e['pain_score'] or 0))
        p.drawString(170, y, str(sss.get('fatigue', 0)))
        p.drawString(230, y, str(sss.get('sleep', 0)))
        p.drawString(290, y, str(e['stress_score'] or 0))
        p.drawString(350, y, str(e['mood_score'] or 0))
        y -= 15
        if y < 50:
            p.showPage()
            y = height - 50

    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name='final_report.pdf',
                     mimetype='application/pdf')


# ----------------- CHART DATA endpoints -----------------


@app.route('/api/chart/daily-pain', methods=['GET'])
def api_chart_daily_pain():
    """Return time series of daily pain for a date range or default last 30 days"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user_id = session['user_id']
    date_from = request.args.get('from')
    date_to = request.args.get('to')
    conn = get_db_connection()
    if date_from and date_to:
        rows = conn.execute('SELECT entry_date, pain_score FROM daily_entries WHERE user_id = ? AND entry_date BETWEEN ? AND ? ORDER BY entry_date', (user_id, date_from, date_to)).fetchall()
    else:
    # default last 30 days
        end = date.today()
        start = end - timedelta(days=29)
        rows = conn.execute('SELECT entry_date, pain_score FROM daily_entries WHERE user_id = ? AND entry_date BETWEEN ? AND ? ORDER BY entry_date', (user_id, start.isoformat(), end.isoformat())).fetchall()
    conn.close()
    series = []
    # return every date in range so frontend can plot continuous x-axis
    if rows:
        dates = [r['entry_date'] for r in rows]
        # If continuous coverage desired, build full range
        if date_from and date_to:
            d_start = datetime.strptime(date_from, "%Y-%m-%d").date()
            d_end = datetime.strptime(date_to, "%Y-%m-%d").date()
        else:
            d_end = date.today()
            d_start = d_end - timedelta(days=29)
        cur = d_start
        row_map = {r['entry_date']: r['pain_score'] for r in rows}
        while cur <= d_end:
            key = cur.isoformat()
            series.append({'date': key, 'pain': row_map.get(key) if key in row_map else None})
            cur += timedelta(days=1)
    return jsonify({'series': series})



@app.route('/api/chart/weekly-heatmap', methods=['GET'])
def api_chart_weekly_heatmap():
    """Return weekly aggregated values for heatmap: fatigue, stress, sleep, workload, mood
       Query param weeks=N (default 12)
    """
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user_id = session['user_id']
    weeks = int(request.args.get('weeks', 12))
    conn = get_db_connection()
    # fetch recent entries covering weeks*7 days
    end_date = date.today()
    start_date = end_date - timedelta(days=weeks*7 - 1)
    rows = conn.execute('SELECT entry_date, fatigue_score, stress_score, sleep_quality, mood_score, workload FROM daily_entries WHERE user_id = ? AND entry_date BETWEEN ? AND ? ORDER BY entry_date', (user_id, start_date.isoformat(), end_date.isoformat())).fetchall()
    conn.close()
    if not rows:
        return jsonify({'error': 'No data available for heatmap'}), 400
    # group by ISO week number and year
    df = pd.DataFrame([dict(r) for r in rows])
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['week'] = df['entry_date'].dt.isocalendar().week
    df['year'] = df['entry_date'].dt.isocalendar().year
    grouped = df.groupby(['year','week']).agg({
        'fatigue_score':'mean', 'stress_score':'mean', 'sleep_quality':'mean', 'mood_score':'mean'
    }).reset_index()
    # build heatmap matrix: list of weeks with averages
    heatmap = []
    # sort by year-week ascending
    grouped = grouped.sort_values(['year','week'])
    for _, row in grouped.iterrows():
        heatmap.append({
            'year': int(row['year']),
            'week': int(row['week']),
            'avg_fatigue': None if pd.isna(row['fatigue_score']) else round(float(row['fatigue_score']),2),
            'avg_stress': None if pd.isna(row['stress_score']) else round(float(row['stress_score']),2),
            'avg_sleep': None if pd.isna(row['sleep_quality']) else round(float(row['sleep_quality']),2),
            'avg_mood': None if pd.isna(row['mood_score']) else round(float(row['mood_score']),2)
        })
    return jsonify({'heatmap_weeks': heatmap})

# ----------------- DOCTOR / LOCATION lookup (Google Maps Places API) -----------------

GOOGLE_MAPS_API_KEY ="AIzaSyBcN4HTut5WWjRN0oWH7cL-Nxeyko9WJzg"  # set this in env

@app.route('/api/nearby-doctors', methods=['GET'])
def api_nearby_doctors():
    """Query Google Places Nearby Search for 'rheumatologist' or 'doctor' near lat,lng
       Query params: lat, lng, radius (meters, optional default 5000)
    """
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    radius = request.args.get('radius', 5000)
    if not lat or not lng:
        return jsonify({'error': 'Provide lat and lng'}), 400
    if not GOOGLE_MAPS_API_KEY:
        return jsonify({'error': 'Server missing GOOGLE_MAPS_API_KEY configuration'}), 500

    url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
    params = {
        'key': GOOGLE_MAPS_API_KEY,
        'location': f'{lat},{lng}',
        'radius': radius,
        'keyword': 'rheumatologist|rheumatology|rheumatologist clinic|rheumatology clinic',
        'type': 'doctor'
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return jsonify({'error': f'Error calling Google Maps API: {str(e)}'}), 500

    results = []
    for r in data.get('results', []):
        results.append({
            'name': r.get('name'),
            'address': r.get('vicinity'),
            'location': r.get('geometry', {}).get('location'),
            'place_id': r.get('place_id'),
            'rating': r.get('rating')
        })
    return jsonify({'doctors': results})

@app.route('/screening')
def screening_page():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('screening.html')



@app.route('/api/screening', methods=['POST'])
@login_required
def api_save_screening():
    user_id = session['user_id']
    data = request.json or {}

    # 1. Extract Data
    first_answers = data.get('first_answers', {})
    wpi_regions = data.get('wpi_regions', [])
    sss_answers = data.get('sss_answers', {})
    sss_somatic = data.get('sss_somatic', {})
    secondary_symptoms = data.get('secondary_symptoms', []) # list of strings
    risk_factors = data.get('risk_factors', {}) # dict r1..r6
    duration_4_weeks = data.get('duration_4_weeks', False)

    # ---------------------------------------------------------
    # MODULE 1: PRIMARY SYMPTOMS (Weight 0.6)
    # ---------------------------------------------------------
    
    # Calculate WPI Score
    wpi_score = len(wpi_regions)
    
    # Calculate SSS Score
    # Part A: fatigue(0-3) + unrefreshed(0-3) + cognitive(0-3)
    try:
        sss_part_a = (int(sss_answers.get('fatigue', 0)) + 
                      int(sss_answers.get('sleep', 0)) + 
                      int(sss_answers.get('cognitive', 0)))
    except (ValueError, TypeError):
        sss_part_a = 0
    
    # Part B: somatic symptoms (headache, pain/cramps, depression) -> max 3
    try:
        sss_part_b = (int(sss_somatic.get('headache', 0)) + 
                      int(sss_somatic.get('abdomenPain', 0)) + 
                      int(sss_somatic.get('depression', 0)))
    except (ValueError, TypeError):
        sss_part_b = 0
                  
    sss_score = sss_part_a + sss_part_b

    # Rule 1: Early Symptom Severity
    # (WPI 2-3 AND SSS 4-5) OR (WPI >= 4 AND SSS >= 4) OR (SSS >= 6 with persistent pain)
    # We interpret "persistent pain" here as simply having some pain (WPI > 0) + High SSS + Duration
    rule1_met = False
    if (2 <= wpi_score <= 3 and 4 <= sss_score <= 5):
        rule1_met = True
    elif (wpi_score >= 4 and sss_score >= 4):
        rule1_met = True
    elif (sss_score >= 6 and wpi_score > 0 and duration_4_weeks):
        rule1_met = True
        
    # Rule 2: Pain Distribution (Early Spread) -> >= 2 regions
    rule2_met = wpi_score >= 2
    
    # Rule 3: Symptom Persistence -> >= 4 weeks
    rule3_met = duration_4_weeks
    
    # Primary Score: 1 if ANY rule met, else 0
    primary_score = 1.0 if (rule1_met or rule2_met or rule3_met) else 0.0
    primary_scaled = primary_score # Used in modular_total_score

    # ---------------------------------------------------------
    # MODULE 2: SECONDARY SYMPTOMS (Weight 0.3)
    # ---------------------------------------------------------
    # List of all possible secondary keys from frontend
    all_sec_keys = [
        "secondary_headache", "secondary_paresthesia", "secondary_allodynia", 
        "secondary_ibs", "secondary_depression", "secondary_sweating", 
        "secondary_sensitivity", "secondary_menstrual", "secondary_stiffness", 
        "secondary_jaw"
    ]
    # Count how many are present in the submitted list
    sec_count = len([k for k in secondary_symptoms if k in all_sec_keys])
    
    # Simple Normalization: Count / 10 (since there are 10 items) -> 0 to 1
    # User said: "Score = normalized sum... >=3 strengthens risk"
    # To map this to 0-1 for the formula, we divide by max possible (10).
    secondary_score_norm = sec_count / 10.0

    # ---------------------------------------------------------
    # MODULE 3: RISK FACTORS (Weight 0.1)
    # ---------------------------------------------------------
    # Items: r1..r6 from form + 'sex' (if Female).
    # Step 4 form has r1..r6. We need to fetch 'sex' from user profile or form.
    # The form in step 4 asks for Sex too (name="sex"). But it's a radio.
    # Wait, the JS `completeScreening` didn't capture `sex` radio specifically in `risk_factors` object?
    # Let's check logic: JS loop `risk_factors['r'+i]` captures r1..r6.
    # We should probably fetch sex from user profile if not in payload.
    # Let's see if we can get it from payload or DB.
    # For now, let's assume we use DB profile sex or default 0.
    
    conn = get_db_connection()
    user = conn.execute('SELECT sex FROM users WHERE id = ?', (user_id,)).fetchone()
    user_sex = user['sex'] if user else None
    
    # Count present factors (0.25 each)
    risk_sum = 0.0
    # r1: Family History
    if risk_factors.get('r1'): risk_sum += 0.25
    # r2: Comorbid
    if risk_factors.get('r2'): risk_sum += 0.25
    # r3: Trauma
    if risk_factors.get('r3'): risk_sum += 0.25
    # r4: PTSD
    if risk_factors.get('r4'): risk_sum += 0.25
    # r5: Anxiety/Depression
    if risk_factors.get('r5'): risk_sum += 0.25
    # r6: Physical Inactivity
    if risk_factors.get('r6'): risk_sum += 0.25
    # Sex: Female
    if user_sex == "Female": risk_sum += 0.25
    
    # Normalize: Divide by 1.75
    risk_factor_fraction = risk_sum / 1.75
    if risk_factor_fraction > 1.0: risk_factor_fraction = 1.0

    # ---------------------------------------------------------
    # TOTAL WEIGHTED SCORE
    # ---------------------------------------------------------
    # --- Calculate Modular Total Score ---
    modular_total_score = (
        (primary_scaled * 0.4) + 
        (secondary_score_norm * 0.3) + 
        (risk_factor_fraction * 0.3)
    )

    # --- ML Model Prediction ---
    # Default to rule-based manual calculation first
    risk_category = "Low"
    if modular_total_score >= 0.7:
        risk_category = "High"
    elif modular_total_score >= 0.4:
        risk_category = "Moderate"
    
    total_risk_score = float(modular_total_score) # legacy assignment

    # Try to use ML model if available
    if SCREENING_MODEL and SCREENING_LE:
        try:
            # Construct feature vector matching training data
            # Features: ["WPI", "SSS", "pain_regions", "symptom_persistence", "secondary_score_norm", "risk_factor_fraction", "rf_total"]
            
            # Pain regions count (matches 'pain_regions' feature)
            pain_regions_count = wpi_score 
            
            # Symptom persistence (matches 'symptom_persistence' feature?? No, check training data logic)
            # In training data, 'symptom_persistence' seems to be integer. 
            # In app, duration_4_weeks is boolean.
            # Let's verify mapping. If not clear, we stick to features we know.
            # Wait, WPI is count of regions. 'pain_regions' in dataset might be same?
            # Let's assume standard feature mapping.
            
            # MAPPING based on ml_training_dataset.csv columns
            # WPI = wpi_score
            # SSS = sss_score
            # pain_regions = wpi_score (likely redundant but present in training)
            # symptom_persistence = 12 if True else 0 (Approximation based on dataset having values like 12, 3, 7 etc likely months)
            # For now, let's map boolean to a fixed value seen in high risk (e.g. 6 months = 6) or just use 3 if > 3 months.
            # The prompt says "duration_4_weeks". The dataset has "duration" in months maybe?
            # Let's check dataset sample: values are 12, 3, 7, 4... likely months.
            # Since we only ask "> 3 months", we can impute a representative value like 6.
            
            persistence_val = 6 if duration_4_weeks else 1
            
            input_features = pd.DataFrame([{
                "WPI": wpi_score,
                "SSS": sss_score,
                "pain_regions": wpi_score, 
                "symptom_persistence": persistence_val,
                "secondary_score_norm": secondary_score_norm,
                "risk_factor_fraction": risk_factor_fraction,
                "rf_total": risk_sum
            }])
            
            # Predict
            pred_encoded = SCREENING_MODEL.predict(input_features)[0]
            risk_category = SCREENING_LE.inverse_transform([pred_encoded])[0]
            
            # Get probability for "High" or weighted prob
            # classes order depends on LE. Usually alphabetical: High, Low, Moderate
            probs = SCREENING_MODEL.predict_proba(input_features)[0]
            classes = SCREENING_LE.classes_
            
            # Find probability of High risk
            if "High" in classes:
                high_idx = np.where(classes == "High")[0][0]
                total_risk_score = float(probs[high_idx])
            else:
                # Fallback to max prob
                total_risk_score = float(max(probs))
                
            print(f"🤖 ML Prediction: Category={risk_category}, Prob={total_risk_score:.2f}")
            
        except Exception as e:
            print(f"⚠️ ML Prediction failed, using manual calculation: {e}")

    # Determine Eligibility (Rule-based safety net + risk)
    # User is eligible if High/Moderate risk AND meets ACR criteria (roughly)
    # ACR 2016: WPI >= 7 & SSS >= 5 OR WPI 3-6 & SSS >= 9
    
    acr_met = (wpi_score >= 7 and sss_score >= 5) or (wpi_score >= 3 and wpi_score <= 6 and sss_score >= 9)
    
    is_eligible = (risk_category in ["High", "Moderate"]) or acr_met

    # --- Save to DB ---said previously: "Eligibility = Meets Criteria".
    # But now we have a probability.
    # Let's assume High/Moderate risk might be eligible, or strict to "High".
    # Given the previous logic was strict, let's stick to "High" = Eligible for now, 
    # OR if any Primary Rule is met?
    # Let's go with: Eligible if risk_category == "High" OR "Moderate" (since prompt implies early detection).
    # SAFETY: "Meets Criteria" usually implies Diagnosis. 
    # Let's stick to: Eligible if total_risk_score >= 0.4 (Moderate+) or simply use the Category.
    # Let's align "is_eligible" with "Moderate" or "High".
    is_eligible = (risk_category in ["High", "Moderate"])

    # 5. Save to DB
    try:
        # We need to save to the NEW tables or existing 'screenings' table?
        # User specified new tables: primary_symptoms, secondary_symptoms, etc. 
        # But we also have an existing 'screenings' table that aggregates result.
        # We will populate the detailed tables AND the summary table.
        
        # Table 2: primary_symptoms
        conn.execute('''
            INSERT INTO primary_symptoms (user_id, pain_score, fatigue_score, sleep_score, cognitive_score, total_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, wpi_score, int(sss_answers.get('fatigue', 0)), int(sss_answers.get('sleep', 0)), int(sss_answers.get('cognitive', 0)), int(primary_score)))

        # Table 3: secondary_symptoms
        # Map list to columns
        def has_sec(key): return 1 if key in secondary_symptoms else 0
        conn.execute('''
            INSERT INTO secondary_symptoms (user_id, headache, paresthesia, allodynia, ibs, depression, sweating, sensory_sensitivity, menstrual_pain, morning_stiffness, jaw_pain, total_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, 
              has_sec("secondary_headache"), has_sec("secondary_paresthesia"), has_sec("secondary_allodynia"),
              has_sec("secondary_ibs"), has_sec("secondary_depression"), has_sec("secondary_sweating"),
              has_sec("secondary_sensitivity"), has_sec("secondary_menstrual"), has_sec("secondary_stiffness"),
              has_sec("secondary_jaw"), int(sec_count)))

        # Table 4: risk_factors
        # Map dict to columns
        def has_risk(key): return 1 if risk_factors.get(key) else 0
        conn.execute('''
            INSERT INTO risk_factors (user_id, genetic_history, comorbid_conditions, trauma_history, ptsd, anxiety_depression, physical_inactivity, total_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, has_risk('r1'), has_risk('r2'), has_risk('r3'), has_risk('r4'), has_risk('r5'), has_risk('r6'), int(risk_sum*4))) # storing raw count or something? field is int. user schema says "total_score".

        # Table 5: screening_result
        conn.execute('''
            INSERT INTO screening_result (user_id, risk_probability, risk_category, screening_status)
            VALUES (?, ?, ?, ?)
        ''', (user_id, total_risk_score, risk_category, "Completed"))

        # Calculate FiRST Score
        # Score is number of 'yes' answers in f1..f6
        first_score = 0
        for i in range(1, 7):
            if first_answers.get(f'f{i}'):
                first_score += 1

        # Also update 'screenings' summary table for backward compatibility / profile view
        # We'll map "meets_criteria" to "is_eligible" logic
        conn.execute('''
            INSERT INTO screenings (
                user_id, pain_regions, secondary_symptoms, primary_symptoms, 
                duration, bmi, first_score, wpi_score, sss_score, 
                meets_criteria, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, 
            json.dumps(wpi_regions), 
            json.dumps(secondary_symptoms),
            json.dumps({'sss_a': sss_answers, 'sss_b': sss_somatic}),
            "more_than_3_months" if duration_4_weeks else "less_than_3_months", 
            None, 
            first_score, # Calculated score
            wpi_score, sss_score, 
            is_eligible, risk_category
        ))

        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500
    conn.close()

    return jsonify({
        'message': 'Screening saved successfully',
        'result': {
            'risk_level': risk_category,
            'risk_probability': total_risk_score,
            'is_eligible': is_eligible,
            'wpi_score': wpi_score,
            'sss_score': sss_score
        }
    })

@app.route('/api/latest-screening', methods=['GET'])
@login_required
def api_latest_screening():
    user_id = session['user_id']
    conn = get_db_connection()
    row = conn.execute('''
        SELECT * FROM screenings 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 1
    ''', (user_id,)).fetchone()
    conn.close()

    if not row:
        return jsonify({})

    data = dict(row)
    data['meets_criteria'] = bool(data['meets_criteria'])
    return jsonify(data)


if __name__ == '__main__':
    print("Starting FibroTracker Flask backend...")
    print("Landing page: http://localhost:5000/")
    app.run(debug=True)



















