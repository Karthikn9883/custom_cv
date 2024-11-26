# backend/create_tables.py

import sqlite3
import os

def create_tables():
    db_path = os.getenv('DB_PATH', 'database/smart_building.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create workers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workers (
            worker_id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_name TEXT NOT NULL,
            worker_number TEXT NOT NULL,
            worker_email TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'free'
        );
    ''')

    # Create tokens table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tokens (
            token_id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_reason TEXT NOT NULL,
            token_location TEXT NOT NULL,
            token_assigned INTEGER,
            token_status TEXT NOT NULL,
            token_start TEXT NOT NULL,
            token_end_time TEXT,
            confidence REAL,
            FOREIGN KEY (token_assigned) REFERENCES workers(worker_id)
        );
    ''')

    # Create cameras table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name TEXT,
            x_coordinate REAL,
            y_coordinate REAL,
            angle REAL,
            room_no TEXT,
            floor INTEGER
        );
    ''')

    # Insert initial data into workers table
    workers = [
        ('Alice Johnson', '555-0101', 'alice@example.com'),
        ('Bob Smith', '555-0102', 'bob@example.com'),
        ('Charlie Davis', '555-0103', 'charlie@example.com'),
    ]

    for worker in workers:
        try:
            cursor.execute('''
                INSERT INTO workers (worker_name, worker_number, worker_email)
                VALUES (?, ?, ?)
            ''', worker)
        except sqlite3.IntegrityError:
            # Worker already exists
            pass

    # Insert example cameras if needed
    cameras = [
        ('Entrance Camera', 100, 150, 0, 'Entrance Hall', 1),
        ('Lobby Camera', 300, 200, 90, 'Lobby', 1),
        ('Hallway Camera', 500, 250, 180, 'Main Hallway', 2),
    ]

    for camera in cameras:
        try:
            cursor.execute('''
                INSERT INTO cameras (camera_name, x_coordinate, y_coordinate, angle, room_no, floor)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', camera)
        except sqlite3.IntegrityError:
            # Camera already exists
            pass

    conn.commit()
    conn.close()
    print("Database tables created and initial data inserted.")

if __name__ == "__main__":
    create_tables()
