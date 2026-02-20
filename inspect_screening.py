import sqlite3
import json

def inspect_kaviyaa():
    conn = sqlite3.connect('fibrotracker.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Try finding 'kaviyaa' or 'user1' based on screenshot cues
    # Screenshot header says "kaviyaa", below that "user1". Maybe username vs display name?
    # Let's search for both.
    
    usernames = ['kaviyaa', 'user1']
    found = False
    
    for name in usernames:
        user = cursor.execute("SELECT id, username FROM users WHERE username = ?", (name,)).fetchone()
        if user:
            print(f"\nUser Found: {user['username']} (ID: {user['id']})")
            screening = cursor.execute("""
                SELECT * FROM screenings 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (user['id'],)).fetchone()
            
            if screening:
                print(f"  Risk Level: {screening['risk_level']}")
                print(f"  FiRST Score: {screening['first_score']}/6")
                print(f"  WPI Score: {screening['wpi_score']}")
                print(f"  SSS Score: {screening['sss_score']}")
                
                # Check detailed symptoms if needed to explain "High Risk"
                primary = json.loads(screening['primary_symptoms']) if screening['primary_symptoms'] else {}
                # primary might be list or dict based on saved format
                
                print(f"  Meets Criteria: {screening['meets_criteria']}")
                found = True
            else:
                print("  No screening found for this user.")
    
    if not found:
        print("Neither 'kaviyaa' nor 'user1' found in users table.")
        # List all just in case
        all_users = cursor.execute("SELECT username FROM users").fetchall()
        print(f"Available users: {[u['username'] for u in all_users]}")

    conn.close()

if __name__ == "__main__":
    inspect_kaviyaa()
