# # backend/create_tables.py

# import sqlite3
# import os

# def create_tables():
#     db_path = os.getenv('DB_PATH', 'database/smart_building.db')
#     os.makedirs(os.path.dirname(db_path), exist_ok=True)

#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # Create workers table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS workers (
#             worker_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             worker_name TEXT NOT NULL,
#             worker_number TEXT NOT NULL,
#             worker_email TEXT NOT NULL UNIQUE,
#             status TEXT NOT NULL DEFAULT 'free'
#         );
#     ''')

#     # Create tokens table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS tokens (
#             token_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             token_reason TEXT NOT NULL,
#             token_location TEXT NOT NULL,
#             token_assigned INTEGER,
#             token_status TEXT NOT NULL,
#             token_start TEXT NOT NULL,
#             token_end_time TEXT,
#             confidence REAL,
#             FOREIGN KEY (token_assigned) REFERENCES workers(worker_id)
#         );
#     ''')

#     # Create cameras table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS cameras (
#             camera_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             camera_name TEXT,
#             x_coordinate REAL,
#             y_coordinate REAL,
#             angle REAL,
#             room_no TEXT,
#             floor INTEGER
#         );
#     ''')

#     # Insert initial data into workers table
#     workers = [
#         ('Alice Johnson', '555-0101', 'karthik.nu9999@gmail.com'),
#         ('Bob Smith', '555-0102', 'karthikn9883@gmail.com'),
#         ('Charlie Davis', '555-0103', 'saidhanushpotluri122@gmail.com'),
#     ]

#     for worker in workers:
#         try:
#             cursor.execute('''
#                 INSERT INTO workers (worker_name, worker_number, worker_email)
#                 VALUES (?, ?, ?)
#             ''', worker)
#         except sqlite3.IntegrityError:
#             # Worker already exists
#             pass

#     # Insert example cameras if needed
#     cameras = [
#         ('Entrance Camera', 100, 150, 0, 'Entrance Hall', 1),
#         ('Lobby Camera', 300, 200, 90, 'Lobby', 1),
#         ('Hallway Camera', 500, 250, 180, 'Main Hallway', 2),
#     ]

#     for camera in cameras:
#         try:
#             cursor.execute('''
#                 INSERT INTO cameras (camera_name, x_coordinate, y_coordinate, angle, room_no, floor)
#                 VALUES (?, ?, ?, ?, ?, ?)
#             ''', camera)
#         except sqlite3.IntegrityError:
#             # Camera already exists
#             pass

#     conn.commit()
#     conn.close()
#     print("Database tables created and initial data inserted.")

# if __name__ == "__main__":
#     create_tables()


# backend/create_tables.py

import sqlite3
import os

def create_tables():
    db_path = os.getenv('DB_PATH', 'database/smart_building.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1) Create 'workers' table if not exists (without last_assigned).
    #    We'll add last_assigned in step (2) below to handle existing DBs.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workers (
            worker_id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_name TEXT NOT NULL,
            worker_number TEXT NOT NULL,
            worker_email TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'free'
            -- last_assigned will be added by ALTER TABLE if missing
        );
    ''')

    # 2) Add 'last_assigned' column if it doesn't exist
    try:
        cursor.execute("SELECT last_assigned FROM workers LIMIT 1;")
        # If this query works, column already exists
    except sqlite3.OperationalError:
        # Column doesn't exist; add it
        cursor.execute("ALTER TABLE workers ADD COLUMN last_assigned DATETIME;")

    # Create 'tokens' table if not exists
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

    # Create 'cameras' table if not exists
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
        ('Alice Johnson', '555-0101', 'karthik.nu9999@gmail.com'),
        ('Bob Smith', '555-0102', 'karthikn9883@gmail.com'),
        ('Charlie Davis', '555-0103', 'saidhanushpotluri122@gmail.com'),
    ]

    for worker in workers:
        try:
            cursor.execute('''
                INSERT INTO workers (worker_name, worker_number, worker_email)
                VALUES (?, ?, ?)
            ''', worker)
        except sqlite3.IntegrityError:
            # Worker with this email already exists, skip
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
            # Camera might already exist, skip
            pass

    conn.commit()
    conn.close()
    print("Database tables created/updated and initial data inserted.")

if __name__ == "__main__":
    create_tables()