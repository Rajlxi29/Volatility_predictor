import sqlite3

def init_db():
    try:
        conn = sqlite3.connect("investments.db")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS investments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                decision TEXT NOT NULL,
                entry_price REAL,
                entry_date TEXT,
                reasoning TEXT
            )
        """)

    except Exception as e:
        print(f"The db has an issue {e}")
    
    finally:
        if(conn):
            conn.commit()
            conn.close()

init_db()
