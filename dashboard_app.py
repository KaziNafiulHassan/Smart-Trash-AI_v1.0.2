# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os
import socket
from datetime import datetime
from neo4j_manager import get_db_manager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Dashboard Setup
st.title("Smart Trash AI - Analytics Dashboard")

# Display connection information for troubleshooting
with st.expander("Connection Details"):
    neo4j_uri = os.getenv("NEO4J_URI", "Not set")
    # Mask password for security
    password_masked = "********" if os.getenv("NEO4J_PASSWORD") else "Not set"
    
    st.write(f"Neo4j URI: {neo4j_uri}")
    st.write(f"Neo4j Username: {os.getenv('NEO4J_USERNAME', 'Not set')}")
    st.write(f"Neo4j Password: {password_masked}")
    
    # Check if hostname can be resolved
    try:
        hostname = neo4j_uri.split("://")[1].split(":")[0]
        st.write(f"Hostname: {hostname}")
        try:
            socket.gethostbyname(hostname)
            st.success(f"Hostname {hostname} successfully resolved")
        except socket.gaierror:
            st.error(f"Cannot resolve hostname: {hostname}. This may indicate network issues.")
    except Exception as e:
        st.error(f"Error parsing URI: {str(e)}")

# Initialize Neo4j connection
try:
    db_manager = get_db_manager()
    st.success("Connected to Neo4j database")
except Exception as e:
    st.error(f"Failed to connect to Neo4j database: {str(e)}")
    st.error("Please check your .env file and Neo4j AuraDB credentials")
    
    # Show more troubleshooting information
    st.error("""
    Troubleshooting steps:
    1. Make sure your .env file contains correct Neo4j AuraDB credentials
    2. Try using the bolt:// protocol (e.g., bolt://hostname:7687)
    3. Check if your AuraDB instance is active
    4. Verify your network connection allows outbound connections to Neo4j
    """)
    
    if st.button("Retry Connection"):
        st.experimental_rerun()
    
    st.stop()

# Create tabbed interface
tabs = st.tabs([
    "User Analytics", 
    "Sorting Game Analytics", 
    "Image Recognition Analytics",
    "Chat Analytics",
    "Knowledge Graph"
])

# User Analytics Tab
with tabs[0]:
    st.header("User Analytics")
    
    # Get user statistics from Neo4j
    user_stats = db_manager.get_user_stats()
    
    # Display user metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Users", user_stats["total_users"])
    
    # Calculate number of nationalities
    nationalities = set(item["nationality"] for item in user_stats.get("users_by_nationality", []))
    col2.metric("Nationalities", len(nationalities))
    
    # Calculate new users today
    today = datetime.now().date()
    recent_users = user_stats.get("recent_users", [])
    
    # Safely count new users from today
    new_users_today = 0
    for user in recent_users:
        created_at = safe_to_datetime(user.get("created_at"))
        if created_at and created_at.date() == today:
            new_users_today += 1
            
    col3.metric("New Users Today", new_users_today)
    
    # Show users by nationality
    if user_stats.get("users_by_nationality"):
        st.subheader("Users by Nationality")
        natl_df = pd.DataFrame(user_stats["users_by_nationality"])
        
        if not natl_df.empty:
            fig = px.pie(natl_df, values="count", names="nationality", 
                        title="User Distribution by Nationality")
            st.plotly_chart(fig, use_container_width=True)
    
    # Show recent users
    if user_stats.get("recent_users"):
        st.subheader("Recent Users")
        
        # Add proper date formatting
        recent_users_list = []
        for user in user_stats["recent_users"]:
            created_dt = safe_to_datetime(user.get("created_at"))
            created_str = created_dt.strftime("%Y-%m-%d %H:%M:%S") if created_dt else "Unknown"
            recent_users_list.append({
                "Username": user.get("username", "Unknown"),
                "Joined": created_str
            })
            
        recent_df = pd.DataFrame(recent_users_list)
        st.dataframe(recent_df)

# Sorting Game Analytics Tab
with tabs[1]:
    st.header("Sorting Game Analytics")
    
    # Get sorting statistics from Neo4j
    sorting_stats = db_manager.get_sorting_stats()
    
    # Display sorting metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Attempts", sorting_stats["total_attempts"])
    
    if "accuracy" in sorting_stats and sorting_stats["accuracy"]:
        accuracy = sorting_stats["accuracy"].get("accuracy", 0) * 100
        col2.metric("Overall Accuracy", f"{accuracy:.1f}%")
        
        correct = sorting_stats["accuracy"].get("correct_attempts", 0)
        col3.metric("Correct Attempts", correct)
    
    # Show item performance
    if "item_performance" in sorting_stats and sorting_stats["item_performance"]:
        st.subheader("Performance by Waste Item")
        
        # Handle potentially empty data
        if sorting_stats["item_performance"]:
            item_df = pd.DataFrame(sorting_stats["item_performance"])
            
            # Format item names for better display
            if not item_df.empty and "item" in item_df.columns:
                item_df["item"] = item_df["item"].apply(lambda x: x.replace("_", " "))
                
                # Create horizontal bar chart sorted by accuracy
                fig = px.bar(item_df.sort_values("accuracy"), 
                            y="item", x="accuracy", 
                            color="accuracy",
                            orientation='h',
                            color_continuous_scale="RdYlGn",
                            labels={"item": "Waste Item", "accuracy": "Accuracy", "attempts": "Attempts"},
                            hover_data=["attempts", "correct"],
                            title="Sorting Accuracy by Waste Item")
                            
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig.update_traces(texttemplate='%{x:.1%}', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No item performance data available yet.")
    
    # Daily performance chart
    if "daily_performance" in sorting_stats and sorting_stats["daily_performance"]:
        st.subheader("Daily Performance Trend")
        
        daily_data = []
        
        for entry in sorting_stats["daily_performance"]:
            day = safe_to_datetime(entry.get("day"))
            day_str = day.strftime("%Y-%m-%d") if day else "Unknown"
            
            daily_data.append({
                "Date": day_str,
                "Attempts": entry.get("attempts", 0),
                "Accuracy": entry.get("accuracy", 0) * 100,  # Convert to percentage
            })
            
        if daily_data:
            daily_df = pd.DataFrame(daily_data)
            
            # Create line chart for daily accuracy
            fig = px.line(daily_df, 
                        x="Date", 
                        y="Accuracy",
                        markers=True,
                        labels={"Date": "Date", "Accuracy": "Accuracy (%)"},
                        title="Daily Sorting Accuracy")
                        
            fig.update_layout(yaxis_range=[0, 100])
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily performance data available yet.")

# Image Recognition Analytics Tab
with tabs[2]:
    st.header("Image Recognition Analytics")
    
    # Get image recognition statistics from Neo4j
    image_stats = db_manager.get_image_recognition_stats()
    
    # Display image metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Uploads", image_stats.get("total_uploads", 0))
    
    if "agreement_rate" in image_stats and image_stats["agreement_rate"]:
        agreements = image_stats["agreement_rate"].get("agreements", 0)
        agreement_rate = image_stats["agreement_rate"].get("agreement_rate", 0) * 100
        
        col2.metric("AI-User Agreements", agreements)
        col3.metric("Agreement Rate", f"{agreement_rate:.1f}%")
    
    # Show bin agreement rate
    if "bin_agreement" in image_stats and image_stats["bin_agreement"]:
        st.subheader("AI Prediction Acceptance Rate by Bin Type")
        
        bin_data = []
        for entry in image_stats["bin_agreement"]:
            bin_data.append({
                "Bin": entry.get("bin", "Unknown").replace("_", " "),
                "Predictions": entry.get("predictions", 0),
                "Agreements": entry.get("agreements", 0),
                "Agreement Rate": entry.get("agreement_rate", 0) * 100
            })
            
        if bin_data:
            bin_df = pd.DataFrame(bin_data)
            
            # Create horizontal bar chart for bin agreement rates
            fig = px.bar(bin_df, 
                        y="Bin", 
                        x="Agreement Rate",
                        color="Agreement Rate",
                        orientation='h',
                        color_continuous_scale="RdYlGn",
                        hover_data=["Predictions", "Agreements"],
                        labels={"Agreement Rate": "Agreement Rate (%)", "Bin": "Bin Type"},
                        title="User Agreement with AI by Bin Type")
                        
            fig.update_layout(xaxis_range=[0, 100])
            fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No bin agreement data available yet.")

# Chat Analytics Tab
with tabs[3]:
    st.header("Chat Analytics")
    
    # Get chat statistics from Neo4j
    chat_stats = db_manager.get_chat_stats()
    
    # Display chat metrics
    st.metric("Total Chat Interactions", chat_stats.get("total_chats", 0))
    
    # Daily chat activity
    if "daily_chats" in chat_stats and chat_stats["daily_chats"]:
        chat_data = []
        
        for entry in chat_stats["daily_chats"]:
            day = safe_to_datetime(entry.get("day"))
            day_str = day.strftime("%Y-%m-%d") if day else "Unknown"
            
            chat_data.append({
                "Date": day_str, 
                "Chats": entry.get("chats", 0)
            })
            
        if chat_data:
            chat_df = pd.DataFrame(chat_data)
            
            # Create bar chart for daily chat counts
            fig = px.bar(chat_df, 
                        x="Date", 
                        y="Chats",
                        title="Daily Chat Interactions")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily chat data available yet.")
    
    # Display top items mentioned in chats
    if "top_items" in chat_stats and chat_stats["top_items"]:
        top_items_data = []
        
        for item in chat_stats["top_items"]:
            top_items_data.append({
                "item": item.get("item", "Unknown").replace("_", " "),
                "mentions": item.get("mentions", 0)
            })
            
        if top_items_data:
            top_items_df = pd.DataFrame(top_items_data)
            
            if not top_items_df.empty:
                # Create a horizontal bar chart for top mentioned items
                top_fig = px.bar(top_items_df, 
                                y='item', 
                                x='mentions', 
                                title="Top Waste Items Mentioned in Chat",
                                labels={"item": "Waste Item", "mentions": "Number of Mentions"},
                                orientation='h')
                top_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(top_fig, use_container_width=True)
        else:
            st.info("No data on items mentioned in chats yet.")

# Knowledge Graph Tab
with tabs[4]:
    st.header("Knowledge Graph Visualization")
    
    # Get knowledge graph data from Neo4j
    try:
        graph_data = db_manager.get_knowledge_graph_data()
        
        if graph_data.get("nodes") and graph_data.get("links"):
            # Create a Networkx graph from the Neo4j data
            G = nx.DiGraph()
            
            # Add nodes with attributes
            for node in graph_data["nodes"]:
                if node and "id" in node and "label" in node:
                    G.add_node(node["id"], label=node["label"], **node.get("properties", {}))
            
            # Add edges with attributes
            for link in graph_data.get("links", []):
                if link and "source" in link and "target" in link:
                    G.add_edge(link["source"], link["target"], label=link.get("type", ""), 
                               **link.get("properties", {}))
            
            if len(G.nodes()) == 0:
                st.info("No nodes found in the knowledge graph. Try using the app more to generate data.")
            else:
                # Create a layout for the graph
                try:
                    pos = nx.spring_layout(G, k=0.15, iterations=50)
                except Exception as e:
                    st.warning(f"Could not create optimal layout: {str(e)}")
                    pos = {node: (i, i) for i, node in enumerate(G.nodes())}
                
                # Create edge traces
                edge_traces = []
                for edge in G.edges():
                    try:
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_trace = go.Scatter(
                            x=[x0, x1, None], y=[y0, y1, None],
                            mode='lines',
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none'
                        )
                        edge_traces.append(edge_trace)
                    except Exception as e:
                        continue
                
                # Create node traces by node type
                node_traces = {}
                node_colors = {
                    'User': '#FF5733',
                    'WasteItem': '#33FF57',
                    'Bin': '#3357FF',
                    'SortingAttempt': '#FF33F5',
                    'ChatInteraction': '#33FFF5',
                    'ImageRecognition': '#F5FF33'
                }
                
                for node in G.nodes():
                    try:
                        node_label = G.nodes[node].get('label', 'Unknown')
                        if node_label not in node_traces:
                            node_traces[node_label] = go.Scatter(
                                x=[], y=[],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(
                                    color=node_colors.get(node_label, '#888888'),
                                    size=10
                                ),
                                name=node_label
                            )
                        
                        x, y = pos[node]
                        node_traces[node_label]['x'] = node_traces[node_label]['x'] + (x,)
                        node_traces[node_label]['y'] = node_traces[node_label]['y'] + (y,)
                        
                        # Create hover text
                        properties = G.nodes[node]
                        hover_text = f"Type: {node_label}<br>"
                        for key, value in properties.items():
                            if key != 'label' and str(value).strip():
                                hover_text += f"{key}: {value}<br>"
                        
                        if hasattr(node_traces[node_label], 'text'):
                            node_traces[node_label]['text'] = node_traces[node_label]['text'] + (hover_text,)
                        else:
                            node_traces[node_label]['text'] = [hover_text]
                    except Exception as e:
                        continue
                
                # Create the figure
                fig = go.Figure(data=edge_traces + list(node_traces.values()))
                
                fig.update_layout(
                    title='Knowledge Graph Visualization',
                    titlefont_size=16,
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics about the graph
                st.subheader("Graph Statistics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Nodes", len(G.nodes()))
                col2.metric("Total Relationships", len(G.edges()))
                
                # Count nodes by type
                node_counts = {}
                for node in G.nodes():
                    node_label = G.nodes[node].get('label', 'Unknown')
                    node_counts[node_label] = node_counts.get(node_label, 0) + 1
                
                col3.metric("Node Types", len(node_counts))
                
                # Count edge types
                edge_types = set()
                for _, _, data in G.edges(data=True):
                    if 'label' in data:
                        edge_types.add(data['label'])
                
                col4.metric("Relationship Types", len(edge_types))
                
                # Display node counts by type
                st.subheader("Nodes by Type")
                node_count_df = pd.DataFrame([
                    {"Type": node_type, "Count": count}
                    for node_type, count in node_counts.items()
                ])
                
                # Sort by count
                if not node_count_df.empty:
                    node_count_df.sort_values('Count', ascending=False, inplace=True)
                    node_count_fig = px.bar(node_count_df, x='Type', y='Count',
                                            title="Node Count by Type")
                    st.plotly_chart(node_count_fig, use_container_width=True)
        else:
            st.info("No knowledge graph data available yet or not enough data to visualize. Try using the app more to generate data.")
    except Exception as e:
        st.error(f"Error rendering knowledge graph: {str(e)}")
        st.info("This could be due to insufficient data. Try using the app more to generate more connections.")

# User Journey Analysis
st.header("User Journey Analysis")
username = st.text_input("Enter username to analyze their journey")

if username:
    try:
        user_journey = db_manager.get_user_journey(username)
        
        if "error" in user_journey:
            st.error(user_journey["error"])
        elif not user_journey.get("user_info"):
            st.error(f"No data found for user '{username}'")
        else:
            # Display user info
            user_info = user_journey["user_info"]
            created_dt = safe_to_datetime(user_info.get("created_at"))
            created_at = created_dt.strftime("%Y-%m-%d %H:%M:%S") if created_dt else "Unknown"
            
            st.subheader(f"Journey for {user_info.get('username', 'Unknown')}")
            col1, col2 = st.columns(2)
            col1.metric("Nationality", user_info.get('nationality', 'Unknown'))
            col2.metric("Joined On", created_at)
            
            # Display learning progress
            if "learning_progress" in user_journey and user_journey["learning_progress"]:
                st.subheader("Learning Progress")
                
                progress_data = []
                for entry in user_journey["learning_progress"]:
                    progress_data.append({
                        "attempt_number": entry.get("attempt_number", 0),
                        "is_correct": entry.get("is_correct", False),
                        "running_accuracy": entry.get("running_accuracy", 0) * 100  # Convert to percentage
                    })
                
                if progress_data:
                    progress_df = pd.DataFrame(progress_data)
                    
                    if not progress_df.empty:
                        # Plot running accuracy over attempts
                        fig = px.line(progress_df, 
                                    x='attempt_number', 
                                    y='running_accuracy',
                                    title="Learning Curve",
                                    labels={"attempt_number": "Attempt #", "running_accuracy": "Accuracy (%)"})
                        fig.update_layout(yaxis_range=[0, 100])
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No learning progress data yet.")
            
            # Display recent sorting attempts
            if "sorting_attempts" in user_journey and user_journey["sorting_attempts"]:
                st.subheader("Recent Sorting Attempts")
                
                sorting_data = []
                for attempt in user_journey["sorting_attempts"]:
                    timestamp = safe_to_datetime(attempt.get("timestamp"))
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "Unknown"
                    
                    sorting_data.append({
                        "item": attempt.get("item", "Unknown").replace("_", " "),
                        "chosen_bin": attempt.get("chosen_bin", "Unknown").replace("_", " "),
                        "correct_bin": attempt.get("correct_bin", "Unknown").replace("_", " "),
                        "is_correct": "✓" if attempt.get("is_correct") else "✗",
                        "timestamp": timestamp_str
                    })
                
                if sorting_data:
                    sorting_df = pd.DataFrame(sorting_data)
                    st.dataframe(sorting_df)
                else:
                    st.info("No sorting attempts data yet.")
            
            # Display recent chat interactions
            if "chats" in user_journey and user_journey["chats"]:
                st.subheader("Recent Chat Interactions")
                
                chat_data = []
                for chat in user_journey["chats"]:
                    timestamp = safe_to_datetime(chat.get("timestamp"))
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "Unknown"
                    
                    chat_data.append({
                        "user_message": chat.get("user_message", ""),
                        "ai_response": chat.get("ai_response", ""),
                        "about_item": chat.get("about_item", "").replace("_", " ") if chat.get("about_item") else "",
                        "timestamp": timestamp_str
                    })
                
                if chat_data:
                    chats_df = pd.DataFrame(chat_data)
                    st.dataframe(chats_df)
                else:
                    st.info("No chat interactions data yet.")
            else:
                st.info("No chat interactions yet. Try using the AI companion!")
            
            # Display recent image recognitions
            if "image_recognitions" in user_journey and user_journey["image_recognitions"]:
                st.subheader("Recent Image Recognition")
                
                img_data = []
                for img in user_journey["image_recognitions"]:
                    timestamp = safe_to_datetime(img.get("timestamp"))
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "Unknown"
                    
                    img_data.append({
                        "image": img.get("image", "Unknown"),
                        "ai_prediction": img.get("ai_prediction", "").replace("_", " "),
                        "user_choice": img.get("user_choice", "").replace("_", " "),
                        "is_agreement": "✓" if img.get("is_agreement") else "✗",
                        "timestamp": timestamp_str
                    })
                
                if img_data:
                    img_df = pd.DataFrame(img_data)
                    st.dataframe(img_df)
                else:
                    st.info("No image recognition data yet.")
    except Exception as e:
        st.error(f"Error retrieving user journey: {str(e)}")
        st.info("Try refreshing the page or check if the database connection is still active.")

# Footer
st.markdown("---")
st.markdown("Smart Trash AI Analytics Dashboard - Powered by Neo4j Knowledge Graph")