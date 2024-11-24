# create_database.py

import sqlite3
import os

def create_tables():
    # Ensure the database directory exists
    db_dir = 'database'
    os.makedirs(db_dir, exist_ok=True)

    # Connect to SQLite database (or create it if it doesn't exist)
    db_path = os.path.join(db_dir, 'smart_building.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create 'workers' table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workers (
            worker_id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_name TEXT NOT NULL,
            worker_number TEXT UNIQUE NOT NULL,
            worker_email TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'free'
        );
    ''')
    # Status column values: 'free' or 'occupied'

    # Create 'tokens' table with foreign key referencing 'workers'
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tokens (
            token_id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_reason TEXT NOT NULL,
            token_location TEXT NOT NULL,
            token_assigned INTEGER,
            token_status TEXT DEFAULT 'Pending',
            token_start DATETIME DEFAULT CURRENT_TIMESTAMP,
            token_end_time DATETIME,
            confidence REAL,
            FOREIGN KEY (token_assigned) REFERENCES workers(worker_id)
        );
    ''')

    # Insert workers into the 'workers' table
    workers = [
        ('ARSHAD', '1234567890', 'karthikn9883@gmail.com'),
        ('Pavan', '0987654321', 'karthik.nu9999@gmail.com'),
        ('Karthik', '1122334455', 'karthik.nutulapati@gmail.com'),
    ]

    for worker_name, worker_number, worker_email in workers:
        try:
            cursor.execute('''
                INSERT INTO workers (worker_name, worker_number, worker_email)
                VALUES (?, ?, ?)
            ''', (worker_name, worker_number, worker_email))
            print(f"Inserted worker: {worker_name}")
        except sqlite3.IntegrityError as e:
            print(f"Error inserting worker {worker_name}: {e}")

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print("Tables 'workers' and 'tokens' created successfully, and workers added.")

if __name__ == "__main__":
    create_tables()
