import sqlite3
from pathlib import Path

db_path = "outputs/chat.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

char_id = 1
rows = cur.execute("SELECT id, image_path FROM character_reference_images WHERE character_id = ?", (char_id,)).fetchall()

print(f"Character {char_id} has {len(rows)} reference images:")
for row in rows:
    print(f"ID: {row[0]}, Path: {row[1]}")

conn.close()
