import sqlite3
import os
import time

DATABASE = 'fibrotracker.db'

def run_migration():
    if not os.path.exists(DATABASE):
        print(f"Database {DATABASE} not found.")
        return

    conn = sqlite3.connect(DATABASE)
    try:
        # Check if migration is needed
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='primary_symptoms'")
        row = cursor.fetchone()
        
        if row and 'CHECK(pain_score BETWEEN 0 AND 10)' in row[0]:
            print("Found old constraint. Starting migration...")
            
            # Use immediate transaction to acquire lock
            conn.execute("BEGIN IMMEDIATE")
            
            try:
                # 1. Rename old table
                conn.execute("ALTER TABLE primary_symptoms RENAME TO primary_symptoms_old")
                
                # 2. Create new table with relaxed constraint (0-19)
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
                
                # 3. Copy data
                conn.execute('''
                    INSERT INTO primary_symptoms (symptom_id, user_id, pain_score, fatigue_score, sleep_score, cognitive_score, total_score)
                    SELECT symptom_id, user_id, pain_score, fatigue_score, sleep_score, cognitive_score, total_score
                    FROM primary_symptoms_old
                ''')
                
                # 4. Drop old table
                conn.execute("DROP TABLE primary_symptoms_old")
                
                conn.commit()
                print("Migration successful: primary_symptoms table schema updated.")
                
            except Exception as e:
                conn.rollback()
                print(f"Migration failed during transaction: {e}")
                raise
        else:
            print("Migration not needed or already applied.")
            
        # Verification Step
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='primary_symptoms'")
        row = cursor.fetchone()
        if 'CHECK(pain_score BETWEEN 0 AND 19)' in row[0]:
            print("VERIFICATION PASSED: Constraint is 0-19.")
        else:
            print("VERIFICATION FAILED: Constraint mismatch.")
            print(f"Current SQL: {row[0]}")

    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            print("Error: Database is locked by another process (likely the running app). Please stop the app and try again.")
        else:
            print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()
