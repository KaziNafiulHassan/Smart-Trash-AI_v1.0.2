# database_manager.py
import sqlite3
from datetime import datetime

DB_PATH = 'app_data.db'  # or whichever name you prefer

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """Create tables if they don't exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # Table for user info
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                nationality TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        # Table for sorting game logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sorting_game_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                waste_item TEXT,
                chosen_bin TEXT,
                correct_bin TEXT,
                is_correct BOOLEAN,
                timestamp TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Table for chat logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_logs (
                chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                user_message TEXT,
                ai_response TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Table for image recognition logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_recognition_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_name TEXT,
                ai_bin TEXT,
                user_bin TEXT,
                is_agreement BOOLEAN,
                timestamp TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        conn.commit()

def add_new_user(username, nationality):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (username, nationality, created_at)
            VALUES (?, ?, ?)
        ''', (username, nationality, datetime.now()))
        conn.commit()
        return cursor.lastrowid  # Return the newly inserted user_id

def get_user_by_username(username):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        return cursor.fetchone()  # or None

def log_sorting_attempt(user_id, waste_item, chosen_bin, correct_bin, is_correct):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sorting_game_logs (user_id, waste_item, chosen_bin, correct_bin, is_correct, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, waste_item, chosen_bin, correct_bin, is_correct, datetime.now()))
        conn.commit()

def log_chat(user_id, user_message, ai_response):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_logs (user_id, user_message, ai_response, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_id, user_message, ai_response, datetime.now()))
        conn.commit()

def log_image_recognition(user_id, image_name, ai_bin, user_bin, is_agreement):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO image_recognition_logs (user_id, image_name, ai_bin, user_bin, is_agreement, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, image_name, ai_bin, user_bin, is_agreement, datetime.now()))
        conn.commit()
