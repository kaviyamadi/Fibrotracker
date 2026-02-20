import sqlite3
import json

def check_schema_and_data():
    conn = sqlite3.connect('fibrotracker.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("--- SCHEMA CHECK ---")
    cursor.execute("PRAGMA table_info(users)")
    columns = [row['name'] for row in cursor.fetchall()]
    
    if 'comorbidities' in columns:
        print("✅ 'comorbidities' column EXISTS in users table.")
    else:
        print("❌ 'comorbidities' column MISSING from users table.")

    print("\n--- DATA CHECK ---")
    users = cursor.execute("SELECT id, username, comorbidities FROM users").fetchall()
    for u in users:
        val = u['comorbidities']
        print(f"User ID: {u['id']}, Username: {u['username']}, Comorbidities: {repr(val)}")

    conn.close()

if __name__ == "__main__":
    check_schema_and_data()
