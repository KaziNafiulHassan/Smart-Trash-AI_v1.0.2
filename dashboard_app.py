# dashboard_app.py
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from database_manager import DB_PATH
from datetime import datetime

# Set page config as the first command
st.set_page_config(page_title="Smart Trash AI Analytics", layout="wide")

# Inject custom CSS after set_page_config
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: #e0e0e0;
        font-family: 'Orbitron', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
    }
    h1, h2 {
        color: #00d4ff;
        text-shadow: 0 0 10px #00d4ff;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
    }
    .stButton>button {
        background: #00d4ff;
        color: #1a1a2e;
        border-radius: 25px;
        border: none;
        padding: 10px 20px;
        box-shadow: 0 0 10px #00d4ff;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px #00d4ff;
    }
    </style>
""", unsafe_allow_html=True)

# Dashboard Setup
st.title("Smart Trash AI - Analytics Dashboard")

# Database connection
conn = sqlite3.connect(DB_PATH)

# Helper Functions
def fetch_data(query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)

def format_timestamp(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col])
    return df

# User Analytics
st.header("User Analytics")
users_df = fetch_data("SELECT * FROM users")
users_df = format_timestamp(users_df, 'created_at')  # Convert 'created_at' to datetime
total_users = len(users_df)
new_users_today = len(users_df[users_df['created_at'].dt.date == datetime.today().date()])

col1, col2 = st.columns(2)
col1.metric("Total Users", total_users)
col2.metric("New Users Today", new_users_today)

nat_fig = px.pie(users_df, names='nationality', title="User Nationality Distribution")
st.plotly_chart(nat_fig, use_container_width=True)

# Sorting Game Analytics
st.header("Sorting Game Analytics")
sort_logs = format_timestamp(fetch_data("SELECT * FROM sorting_game_logs"), 'timestamp')
total_attempts = len(sort_logs)
accuracy = (sort_logs['is_correct'].sum() / total_attempts * 100) if total_attempts > 0 else 0
st.metric("Total Sorting Attempts", total_attempts)
st.metric("Overall Accuracy", f"{accuracy:.2f}%")

sort_logs['date'] = sort_logs['timestamp'].dt.date
daily_accuracy = sort_logs.groupby('date')['is_correct'].mean().reset_index()
acc_fig = px.line(daily_accuracy, x='date', y='is_correct', title="Daily Sorting Accuracy")
st.plotly_chart(acc_fig, use_container_width=True)

# Image Recognition Analytics
st.header("Image Recognition Analytics")
image_logs = format_timestamp(fetch_data("SELECT * FROM image_recognition_logs"), 'timestamp')
total_images = len(image_logs)
agreement_rate = (image_logs['is_agreement'].sum() / total_images * 100) if total_images > 0 else 0
st.metric("Total Image Uploads", total_images)
st.metric("AI-User Agreement Rate", f"{agreement_rate:.2f}%")

bin_agreement = image_logs.groupby('ai_bin')['is_agreement'].mean().reset_index()
bin_fig = px.bar(bin_agreement, x='ai_bin', y='is_agreement', title="Agreement Rate by Bin")
st.plotly_chart(bin_fig, use_container_width=True)

# Chat Analytics
st.header("Chat Analytics")
chat_logs = format_timestamp(fetch_data("SELECT * FROM chat_logs"), 'timestamp')
total_chats = len(chat_logs)
st.metric("Total Chat Messages", total_chats)

daily_chats = chat_logs.groupby(chat_logs['timestamp'].dt.date).size().reset_index(name='count')
chat_fig = px.line(daily_chats, x='timestamp', y='count', title="Daily Chat Interactions")
st.plotly_chart(chat_fig, use_container_width=True)

# Cleanup
conn.close()