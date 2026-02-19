import sqlite3
import os

# Ensure we are in the right directory
db_path = 'fibrotracker.db'

def verify_constraint():
    conn = sqlite3.connect(db_path)
    try:
        # Check table schema
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='primary_symptoms'")
        row = cursor.fetchone()
        if not row:
            print("❌ Table primary_symptoms not found.")
            return

        print(f"Schema found: {row[0]}")
        
        if 'CHECK(pain_score BETWEEN 0 AND 19)' in row[0]:
            print("✅ Constraint updated successfully to 0-19.")
        else:
            print("❌ Constraint NOT updated.")

        # Try inserting a high value
        print("Attempting to insert pain_score = 15...")
        try:
            conn.execute("INSERT INTO primary_symptoms (user_id, pain_score, fatigue_score, sleep_score, cognitive_score, total_score) VALUES (1, 15, 5, 5, 5, 1)")
            print("✅ Insertion of pain_score 15 succeeded.")
            conn.rollback() # Don't actually save it
        except sqlite3.IntegrityError as e:
            print(f"❌ Insertion failed: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
    else:
        verify_constraint()
