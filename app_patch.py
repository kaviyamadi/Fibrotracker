
# **************** API ENDPOINTS - SCREENING **********************************

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
    secondary_symptoms = data.get('secondary_symptoms', [])
    risk_factors = data.get('risk_factors', {})

    # 2. Calculate Scores
    # FiRST: Count 'Yes' (True)
    first_score = sum(1 for v in first_answers.values() if v)
    
    # WPI: Count regions
    wpi_score = len(wpi_regions)
    
    # SSS
    # Part A: fatigue(0-3) + unrefreshed(0-3) + cognitive(0-3)
    sss_part_a = (int(sss_answers.get('fatigue', 0)) + 
                  int(sss_answers.get('sleep', 0)) + 
                  int(sss_answers.get('cognitive', 0)))
    
    # Part B: somatic symptoms (headache, pain/cramps, depression) - in our UI these are 0 or 1
    # But usually represented as intensity 0-3 in full questionnaires. 
    # Here we sum the booleans/checks. Max 3.
    sss_part_b = (int(sss_somatic.get('headache', 0)) + 
                  int(sss_somatic.get('abdomenPain', 0)) + 
                  int(sss_somatic.get('depression', 0)))
                  
    sss_score = sss_part_a + sss_part_b

    # 3. Diagnostic Criteria (2016)
    # (WPI >= 7 && SSS >= 5) OR (WPI 3-6 && SSS >= 9)
    meets_criteria = False
    if (wpi_score >= 7 and sss_score >= 5) or (3 <= wpi_score <= 6 and sss_score >= 9):
        meets_criteria = True

    # 4. Risk Level
    if meets_criteria:
        risk_level = "High"
    elif first_score >= 5: # FiRST is highly sensitive
        risk_level = "High" 
    elif first_score >= 3 or (wpi_score + sss_score) >= 10: # threshold guess
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # 5. Metadata
    # Duration could be asked in form, or we assume chronic if they use this app
    duration = "3_months_plus" # placeholder
    
    # BMI calculation if we had height/weight, but we don't.
    bmi = None

    # 6. Save to DB
    conn = get_db_connection()
    try:
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
            json.dumps({'first': first_answers, 'sss_a': sss_answers, 'sss_b': sss_somatic}),
            duration, bmi, first_score, wpi_score, sss_score, 
            meets_criteria, risk_level
        ))
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500
    conn.close()

    return jsonify({
        'message': 'Screening saved successfully',
        'result': {
            'risk_level': risk_level,
            'meets_criteria': meets_criteria
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
    # handle boolean
    data['meets_criteria'] = bool(data['meets_criteria'])
    return jsonify(data)
