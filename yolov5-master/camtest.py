import sqlite3

db_path = "database/smart_building.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT * FROM cameras")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()
