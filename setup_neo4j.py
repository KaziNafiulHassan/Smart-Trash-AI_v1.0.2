"""
Setup and Migration Script for Neo4j Database

This script helps migrate data from SQLite to Neo4j and sets up the Neo4j database.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from neo4j_manager import get_db_manager, init_db

# Load environment variables
load_dotenv()

def check_env_variables():
    """Check if Neo4j environment variables are set."""
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Please create a .env file with the required Neo4j credentials.")
        return False
    return True

def migrate_from_sqlite(sqlite_db_path):
    """Migrate data from SQLite to Neo4j."""
    if not os.path.exists(sqlite_db_path):
        print(f"SQLite database not found at {sqlite_db_path}")
        return False
    
    try:
        # Connect to SQLite
        conn = sqlite3.connect(sqlite_db_path)
        
        # Get data from SQLite
        users_df = pd.read_sql("SELECT * FROM users", conn)
        sorting_logs_df = pd.read_sql("SELECT * FROM sorting_game_logs", conn)
        chat_logs_df = pd.read_sql("SELECT * FROM chat_logs", conn)
        image_logs_df = pd.read_sql("SELECT * FROM image_recognition_logs", conn)
        
        # Get Neo4j manager
        db_manager = get_db_manager()
        
        # Import users
        print(f"Importing {len(users_df)} users...")
        for _, user in users_df.iterrows():
            user_id = db_manager.add_user(user['username'], user['nationality'])
            print(f"Added user {user['username']} with ID {user_id}")
        
        # Import sorting logs
        print(f"Importing {len(sorting_logs_df)} sorting logs...")
        for _, log in sorting_logs_df.iterrows():
            # Get user by original SQLite ID
            sqlite_user = users_df[users_df['user_id'] == log['user_id']].iloc[0]
            username = sqlite_user['username']
            
            # Get Neo4j user
            neo4j_user = db_manager.get_user_by_username(username)
            
            if neo4j_user:
                db_manager.log_sorting_attempt(
                    neo4j_user['user_id'],
                    log['waste_item'],
                    log['chosen_bin'],
                    log['correct_bin'],
                    bool(log['is_correct'])
                )
        
        # Import chat logs
        print(f"Importing {len(chat_logs_df)} chat logs...")
        for _, log in chat_logs_df.iterrows():
            # Get user by original SQLite ID
            sqlite_user = users_df[users_df['user_id'] == log['user_id']].iloc[0]
            username = sqlite_user['username']
            
            # Get Neo4j user
            neo4j_user = db_manager.get_user_by_username(username)
            
            if neo4j_user:
                db_manager.log_chat(
                    neo4j_user['user_id'],
                    log['user_message'],
                    log['ai_response']
                )
        
        # Import image recognition logs
        print(f"Importing {len(image_logs_df)} image recognition logs...")
        for _, log in image_logs_df.iterrows():
            # Get user by original SQLite ID
            sqlite_user = users_df[users_df['user_id'] == log['user_id']].iloc[0]
            username = sqlite_user['username']
            
            # Get Neo4j user
            neo4j_user = db_manager.get_user_by_username(username)
            
            if neo4j_user:
                db_manager.log_image_recognition(
                    neo4j_user['user_id'],
                    log['image_name'],
                    log['ai_bin'],
                    log['user_bin'],
                    bool(log['is_agreement'])
                )
        
        print("Migration completed successfully!")
        return True
    
    except Exception as e:
        print(f"Migration failed: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def setup_neo4j():
    """Initialize Neo4j database with schema and constraints."""
    try:
        print("Initializing Neo4j database...")
        init_db()
        print("Neo4j database initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize Neo4j database: {str(e)}")
        return False

def main():
    """Main function to run the setup and migration."""
    print("==== Smart Trash AI: Neo4j Setup and Migration Tool ====")
    
    # Check environment variables
    if not check_env_variables():
        return
    
    # Setup Neo4j
    if not setup_neo4j():
        return
    
    # Ask for migration
    migrate = input("Do you want to migrate data from SQLite? (y/n): ").lower().strip() == 'y'
    
    if migrate:
        sqlite_path = input("Enter the path to SQLite database (default: app_data.db): ").strip()
        if not sqlite_path:
            sqlite_path = "app_data.db"
        
        migrate_from_sqlite(sqlite_path)
    
    print("\nSetup completed!")
    print("\nNEXT STEPS:")
    print("1. Make sure your .env file has the correct Neo4j credentials.")
    print("2. Run the application with: python app.py")
    print("3. Run the dashboard with: streamlit run dashboard_app.py")

if __name__ == "__main__":
    main() 