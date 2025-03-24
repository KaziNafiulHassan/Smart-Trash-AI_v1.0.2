import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from neo4j_manager import get_db_manager
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Helper function to safely convert Neo4j DateTime objects to Python datetime
def safe_to_datetime(value):
    """
    Safely convert various datetime formats to Python datetime objects
    """
    if value is None:
        return None
    
    # If it's already a datetime, return it
    if isinstance(value, datetime):
        return value
    
    # If it's a neo4j.time.DateTime object, convert it to string first
    try:
        if hasattr(value, 'iso_format'):
            value = value.iso_format()
    except:
        pass
    
    # Try converting from string
    try:
        return pd.to_datetime(str(value))
    except:
        return None

# Page config
st.set_page_config(page_title="Smart Trash AI - Simple Dashboard", layout="wide")

# Dashboard Setup
st.title("Smart Trash AI - Simple Analytics Dashboard")

# Display connection information
with st.expander("Connection Details"):
    neo4j_uri = os.getenv("NEO4J_URI", "Not set")
    st.write(f"Neo4j URI: {neo4j_uri}")
    
# Initialize Neo4j connection
try:
    db_manager = get_db_manager()
    st.success("Connected to Neo4j database")
    
    # User Analytics Section
    st.header("User Analytics")
    
    with db_manager.driver.session() as session:
        # User count
        user_count = session.run("MATCH (u:User) RETURN count(u) as count").single()["count"]
        st.metric("Total Users", user_count)
        
        # User list
        users_query = """
        MATCH (u:User) 
        RETURN u.username, u.nationality, u.created_at
        ORDER BY u.created_at DESC
        """
        users = session.run(users_query).data()
        
        if users:
            # Format dates properly using the safe_to_datetime function
            formatted_users = []
            for user in users:
                created_dt = safe_to_datetime(user["u.created_at"])
                created_str = created_dt.strftime("%Y-%m-%d %H:%M:%S") if created_dt else "Unknown"
                
                formatted_users.append({
                    "Username": user["u.username"],
                    "Nationality": user["u.nationality"],
                    "Joined": created_str
                })
            
            df_users = pd.DataFrame(formatted_users)
            
            st.subheader("User List")
            st.dataframe(df_users)
        
        # Sorting Game Analytics
        st.header("Sorting Game Analytics")
        
        # Get total attempts
        attempts_count = session.run("""
            MATCH (a:SortingAttempt)
            RETURN count(a) as count
        """).single()["count"]
        
        # Get correct attempts
        correct_count = session.run("""
            MATCH (a:SortingAttempt)
            WHERE a.is_correct = true
            RETURN count(a) as count
        """).single()["count"]
        
        # Calculate accuracy
        accuracy = (correct_count / attempts_count * 100) if attempts_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Attempts", attempts_count)
        col2.metric("Correct Attempts", correct_count)
        col3.metric("Accuracy", f"{accuracy:.1f}%")
        
        # Recent attempts
        attempts_query = """
        MATCH (u:User)-[:ATTEMPTED]->(a:SortingAttempt)-[:FOR_ITEM]->(w:WasteItem)
        MATCH (a)-[:SELECTED]->(chosen:Bin)
        MATCH (a)-[:SHOULD_SELECT]->(correct:Bin)
        RETURN 
            u.username as username,
            w.name as item,
            chosen.name as chosen_bin,
            correct.name as correct_bin,
            a.is_correct as is_correct,
            a.timestamp as timestamp
        ORDER BY a.timestamp DESC
        LIMIT 20
        """
        
        attempts = session.run(attempts_query).data()
        
        if attempts:
            # Format dates properly
            formatted_attempts = []
            for a in attempts:
                timestamp_dt = safe_to_datetime(a["timestamp"])
                timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S") if timestamp_dt else "Unknown"
                
                formatted_attempts.append({
                    "Username": a["username"],
                    "Item": a["item"].replace("_", " "),
                    "Chosen Bin": a["chosen_bin"].replace("_", " "),
                    "Correct Bin": a["correct_bin"].replace("_", " "),
                    "Result": "✓" if a["is_correct"] else "✗",
                    "Time": timestamp_str
                })
            
            df_attempts = pd.DataFrame(formatted_attempts)
            
            st.subheader("Recent Sorting Attempts")
            st.dataframe(df_attempts)
            
            # Performance by bin type
            bin_performance = session.run("""
            MATCH (a:SortingAttempt)-[:SHOULD_SELECT]->(b:Bin)
            RETURN 
                b.name as bin,
                count(a) as attempts,
                sum(CASE WHEN a.is_correct THEN 1 ELSE 0 END) as correct,
                toFloat(sum(CASE WHEN a.is_correct THEN 1 ELSE 0 END)) / count(a) as accuracy
            ORDER BY attempts DESC
            """).data()
            
            if bin_performance:
                df_bins = pd.DataFrame([{
                    "Bin Type": b["bin"].replace("_", " "),
                    "Attempts": b["attempts"],
                    "Correct": b["correct"],
                    "Accuracy": b["accuracy"] * 100
                } for b in bin_performance])
                
                st.subheader("Performance by Bin Type")
                
                # Bar chart
                fig = px.bar(df_bins, 
                             x="Bin Type", 
                             y="Accuracy", 
                             text_auto='.1f',
                             color="Accuracy",
                             color_continuous_scale="RdYlGn")
                fig.update_layout(yaxis_title="Accuracy (%)")
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Image Recognition Data
        st.header("Image Recognition Analytics")
        
        # Image count
        image_count = session.run("""
            MATCH (img:ImageRecognition)
            RETURN count(img) as count
        """).single()["count"]
        
        # Agreement count
        agreement_count = session.run("""
            MATCH (img:ImageRecognition)
            WHERE img.is_agreement = true
            RETURN count(img) as count
        """).single()["count"]
        
        # Calculate agreement rate
        agreement_rate = (agreement_count / image_count * 100) if image_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Images", image_count)
        col2.metric("AI-User Agreements", agreement_count)
        col3.metric("Agreement Rate", f"{agreement_rate:.1f}%")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Check the console for more details on the error.") 