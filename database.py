import sqlite3
import os

class DatabaseHandler:
    def __init__(self, db_name="Number_Plate_Detection.db"):
        # Current working directory ke sath exact database name set kiya hai
        basedir = os.getcwd()
        self.db_path = os.path.join(basedir, db_name)
        print(f"👉 [DATABASE INFO] Database path is: {self.db_path}")
        self.create_table()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def create_table(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        # Table name explicitly 'license_plates' rakha hai
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS license_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT UNIQUE,
                image_path TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def is_plate_registered(self, plate_number):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM license_plates WHERE plate_number = ?", (plate_number,))
        row = cursor.fetchone()
        conn.close()
        return row is not None

    def insert_plate(self, plate_number, image_path, confidence):
        print(f"💾 [DB INSERT] Saving plate: {plate_number} | Path: {image_path}")
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO license_plates (plate_number, image_path, confidence)
                VALUES (?, ?, ?)
            ''', (plate_number, image_path, confidence))
            conn.commit()
            print("✅ [DB SUCCESS] Record inserted successfully!")
        except sqlite3.IntegrityError:
            print("⚠️ [DB INFO] Plate already exists in database.")
        finally:
            conn.close()

    def fetch_all_plates(self):
        """Fetches all saved plates from 'license_plates' table for Streamlit UI"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, plate_number, image_path, confidence, timestamp FROM license_plates ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def clear_database(self):
        """Clears all records from 'license_plates' table"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM license_plates")
        conn.commit()
        conn.close()