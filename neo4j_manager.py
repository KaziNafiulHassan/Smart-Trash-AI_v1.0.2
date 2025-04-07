"""
Neo4j Database Manager for Smart Trash AI

This module provides the interface for interacting with a Neo4j AuraDB database,
implementing a knowledge graph structure for waste sorting data.
"""

import os
import logging
import socket
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth
import time

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection details (from environment variables)
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

class Neo4jManager:
    """Manager class for Neo4j database operations."""
    
    def __init__(self):
        """Initialize the Neo4j database connection."""
        if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
            logger.error("Neo4j connection details not fully provided in environment variables.")
            raise ValueError("Missing Neo4j connection details. Please check NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")
        
        # Extract hostname for DNS check
        try:
            hostname = NEO4J_URI.split("://")[1].split(":")[0]
            logger.info(f"Attempting to connect to Neo4j at {hostname}")
            
            # Simple DNS check - but don't fail if it doesn't resolve
            try:
                socket.gethostbyname(hostname)
                logger.info(f"Successfully resolved hostname: {hostname}")
            except socket.gaierror as dns_error:
                logger.warning(f"Could not resolve hostname: {hostname}. Error: {str(dns_error)}")
                logger.warning("Continuing anyway as this might be a temporary DNS issue or network configuration")
        except Exception as e:
            logger.warning(f"Error parsing hostname from URI: {str(e)}")
            logger.warning("Continuing with connection attempt anyway")
        
        # Connection retry logic
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to Neo4j with URI: {NEO4J_URI} (Attempt {retry_count + 1}/{max_retries})")
                
                # Create driver with SSL encryption enabled but more permissive settings
                self.driver = GraphDatabase.driver(
                    NEO4J_URI, 
                    auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD),
                    encrypted=True,
                    trust="TRUST_ALL_CERTIFICATES"
                )
                
                # Test connection with simple query and a short timeout
                with self.driver.session() as session:
                    result = session.run("RETURN 'Connected!' AS message")
                    message = result.single()["message"]
                    logger.info(f"Connection test result: {message}")
                    
                logger.info("Successfully connected to Neo4j database")
                return  # Success - exit the method
                
            except Exception as e:
                last_error = e
                retry_count += 1
                logger.warning(f"Connection attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count < max_retries:
                    logger.info(f"Retrying in 2 seconds...")
                    time.sleep(2)  # Wait before retrying
        
        # If we get here, all retries failed
        logger.error(f"Failed to connect to Neo4j after {max_retries} attempts. Last error: {str(last_error)}")
        logger.error("Please check if:")
        logger.error("1. Your Neo4j AuraDB instance is active (not paused)")
        logger.error("2. Credentials in environment variables are correct")
        logger.error("3. Network connectivity to Neo4j is available")
        logger.error("4. Visit Neo4j AuraDB dashboard to ensure instance is running")
        raise last_error
    
    def close(self):
        """Close the Neo4j driver connection."""
        if hasattr(self, 'driver'):
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def init_db(self):
        """Initialize database with constraints and indexes."""
        with self.driver.session() as session:
            # Create constraints for unique nodes
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
            session.run("CREATE CONSTRAINT username IF NOT EXISTS FOR (u:User) REQUIRE u.username IS UNIQUE")
            session.run("CREATE CONSTRAINT waste_item IF NOT EXISTS FOR (w:WasteItem) REQUIRE w.name IS UNIQUE")
            session.run("CREATE CONSTRAINT bin_type IF NOT EXISTS FOR (b:Bin) REQUIRE b.name IS UNIQUE")
            
            logger.info("Database initialized with constraints")
            
            # Initialize waste classification data
            self._init_waste_classification()
    
    def _init_waste_classification(self):
        """Initialize waste classification data from CSV."""
        import pandas as pd
        
        try:
            waste_data = pd.read_csv('waste_classification.csv')
            
            with self.driver.session() as session:
                # Create waste items and bins
                for _, row in waste_data.iterrows():
                    waste_item = row['Waste_Item']
                    container = row['Waste_Containers_Type']
                    bin_color = row['Bin_Color']
                    category = row['Category']
                    
                    # Create waste item node
                    session.run("""
                    MERGE (w:WasteItem {name: $waste_item})
                    SET w.category = $category
                    """, waste_item=waste_item, category=category)
                    
                    # Create bin node
                    session.run("""
                    MERGE (b:Bin {name: $container})
                    SET b.color = $bin_color
                    """, container=container, bin_color=bin_color)
                    
                    # Create relationship between waste item and bin
                    session.run("""
                    MATCH (w:WasteItem {name: $waste_item})
                    MATCH (b:Bin {name: $container})
                    MERGE (w)-[:BELONGS_TO]->(b)
                    """, waste_item=waste_item, container=container)
            
            logger.info(f"Initialized {len(waste_data)} waste classification items")
        except Exception as e:
            logger.error(f"Failed to initialize waste classification data: {str(e)}")
    
    def add_user(self, username: str, nationality: str) -> int:
        """
        Add a new user to the database.
        
        Args:
            username: User's chosen username
            nationality: User's nationality
            
        Returns:
            user_id: The ID of the created user
        """
        with self.driver.session() as session:
            result = session.run("""
            MERGE (u:User {username: $username})
            ON CREATE SET u.user_id = randomUUID(),
                          u.nationality = $nationality,
                          u.created_at = datetime()
            ON MATCH SET u.nationality = $nationality
            RETURN u.user_id AS user_id
            """, username=username, nationality=nationality)
            
            record = result.single()
            if not record:
                raise Exception("Failed to create user")
            
            user_id = record["user_id"]
            logger.info(f"User added/updated: {username} with ID {user_id}")
            return user_id
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user details by username.
        
        Args:
            username: User's username
            
        Returns:
            Dictionary with user details or None if not found
        """
        with self.driver.session() as session:
            result = session.run("""
            MATCH (u:User {username: $username})
            RETURN u.user_id AS user_id, u.username AS username, 
                   u.nationality AS nationality, u.created_at AS created_at
            """, username=username)
            
            record = result.single()
            if not record:
                return None
            
            return {
                "user_id": record["user_id"],
                "username": record["username"],
                "nationality": record["nationality"],
                "created_at": record["created_at"]
            }
    
    def log_sorting_attempt(self, user_id: str, waste_item: str, chosen_bin: str, 
                          correct_bin: str, is_correct: bool):
        """
        Log a waste sorting attempt.
        
        Args:
            user_id: The user's ID
            waste_item: The waste item name
            chosen_bin: The bin chosen by the user
            correct_bin: The correct bin for the item
            is_correct: Whether the choice was correct
        """
        with self.driver.session() as session:
            session.run("""
            MATCH (u:User {user_id: $user_id})
            MATCH (w:WasteItem {name: $waste_item})
            MATCH (chosen:Bin {name: $chosen_bin})
            MATCH (correct:Bin {name: $correct_bin})
            
            CREATE (a:SortingAttempt {
                attempt_id: randomUUID(),
                timestamp: datetime(),
                is_correct: $is_correct
            })
            
            CREATE (u)-[:ATTEMPTED]->(a)
            CREATE (a)-[:SELECTED]->(chosen)
            CREATE (a)-[:SHOULD_SELECT]->(correct)
            CREATE (a)-[:FOR_ITEM]->(w)
            """, user_id=user_id, waste_item=waste_item, chosen_bin=chosen_bin,
                 correct_bin=correct_bin, is_correct=is_correct)
            
            logger.info(f"Logged sorting attempt for user {user_id}, item {waste_item}")
    
    def log_chat(self, user_id: str, user_message: str, ai_response: str):
        """
        Log a chat interaction.
        
        Args:
            user_id: The user's ID
            user_message: The message from the user
            ai_response: The response from the AI
        """
        with self.driver.session() as session:
            session.run("""
            MATCH (u:User {user_id: $user_id})
            
            CREATE (c:ChatInteraction {
                chat_id: randomUUID(),
                user_message: $user_message,
                ai_response: $ai_response,
                timestamp: datetime()
            })
            
            CREATE (u)-[:CHATTED]->(c)
            """, user_id=user_id, user_message=user_message, ai_response=ai_response)
            
            # If the user message contains a waste item name, create relationship to that item
            session.run("""
            MATCH (c:ChatInteraction {user_message: $user_message})
            MATCH (w:WasteItem)
            WHERE toLower(c.user_message) CONTAINS toLower(w.name)
            MERGE (c)-[:ABOUT_ITEM]->(w)
            """, user_message=user_message)
            
            logger.info(f"Logged chat for user {user_id}")
    
    def log_image_recognition(self, user_id: str, image_name: str, ai_bin: str, 
                            user_bin: str, is_agreement: bool):
        """
        Log an image recognition interaction.
        
        Args:
            user_id: The user's ID
            image_name: Name or path of the uploaded image
            ai_bin: The bin predicted by AI
            user_bin: The bin chosen by the user
            is_agreement: Whether the user agreed with AI prediction
        """
        with self.driver.session() as session:
            session.run("""
            MATCH (u:User {user_id: $user_id})
            MATCH (ai:Bin {name: $ai_bin})
            MATCH (user:Bin {name: $user_bin})
            
            CREATE (img:ImageRecognition {
                recognition_id: randomUUID(),
                image_name: $image_name,
                is_agreement: $is_agreement,
                timestamp: datetime()
            })
            
            CREATE (u)-[:UPLOADED]->(img)
            CREATE (img)-[:AI_PREDICTED]->(ai)
            CREATE (img)-[:USER_SELECTED]->(user)
            """, user_id=user_id, image_name=image_name, ai_bin=ai_bin,
                 user_bin=user_bin, is_agreement=is_agreement)
            
            logger.info(f"Logged image recognition for user {user_id}, image {image_name}")
    
    # Analytics functions
    def get_user_stats(self) -> Dict:
        """Get user statistics."""
        with self.driver.session() as session:
            total_users = session.run("""
            MATCH (u:User)
            RETURN count(u) AS total
            """).single()["total"]
            
            users_by_nationality = session.run("""
            MATCH (u:User)
            RETURN u.nationality AS nationality, count(u) AS count
            ORDER BY count DESC
            """).data()
            
            recent_users = session.run("""
            MATCH (u:User)
            RETURN u.username AS username, u.created_at AS created_at
            ORDER BY u.created_at DESC
            LIMIT 10
            """).data()
            
            return {
                "total_users": total_users,
                "users_by_nationality": users_by_nationality,
                "recent_users": recent_users
            }
    
    def get_sorting_stats(self) -> Dict:
        """Get sorting game statistics."""
        with self.driver.session() as session:
            total_attempts = session.run("""
            MATCH (a:SortingAttempt)
            RETURN count(a) AS total
            """).single()["total"]
            
            accuracy = session.run("""
            MATCH (a:SortingAttempt)
            RETURN 
                count(a) AS total_attempts,
                sum(CASE WHEN a.is_correct THEN 1 ELSE 0 END) AS correct_attempts,
                toFloat(sum(CASE WHEN a.is_correct THEN 1 ELSE 0 END)) / 
                    CASE WHEN count(a) > 0 THEN count(a) ELSE 1 END AS accuracy
            """).single()
            
            item_performance = session.run("""
            MATCH (a:SortingAttempt)-[:FOR_ITEM]->(w:WasteItem)
            RETURN 
                w.name AS item,
                count(a) AS attempts,
                sum(CASE WHEN a.is_correct THEN 1 ELSE 0 END) AS correct,
                toFloat(sum(CASE WHEN a.is_correct THEN 1 ELSE 0 END)) / 
                    count(a) AS accuracy
            ORDER BY attempts DESC
            LIMIT 20
            """).data()
            
            # Daily performance over time - Fixed to properly define variable 'a'
            daily_performance = session.run("""
            MATCH (a:SortingAttempt)
            WITH date(a.timestamp) AS day, count(a) AS dayAttempts,
                 sum(CASE WHEN a.is_correct THEN 1 ELSE 0 END) AS dayCorrect
            RETURN 
                day,
                dayAttempts AS attempts,
                dayCorrect AS correct,
                toFloat(dayCorrect) / dayAttempts AS accuracy
            ORDER BY day
            """).data()
            
            return {
                "total_attempts": total_attempts,
                "accuracy": accuracy,
                "item_performance": item_performance,
                "daily_performance": daily_performance
            }
    
    def get_image_recognition_stats(self) -> Dict:
        """Get image recognition statistics."""
        with self.driver.session() as session:
            total_uploads = session.run("""
            MATCH (img:ImageRecognition)
            RETURN count(img) AS total
            """).single()["total"]
            
            agreement_rate = session.run("""
            MATCH (img:ImageRecognition)
            RETURN 
                count(img) AS total,
                sum(CASE WHEN img.is_agreement THEN 1 ELSE 0 END) AS agreements,
                toFloat(sum(CASE WHEN img.is_agreement THEN 1 ELSE 0 END)) / 
                    CASE WHEN count(img) > 0 THEN count(img) ELSE 1 END AS agreement_rate
            """).single()
            
            bin_agreement = session.run("""
            MATCH (img:ImageRecognition)-[:AI_PREDICTED]->(b:Bin)
            RETURN 
                b.name AS bin,
                count(img) AS predictions,
                sum(CASE WHEN img.is_agreement THEN 1 ELSE 0 END) AS agreements,
                toFloat(sum(CASE WHEN img.is_agreement THEN 1 ELSE 0 END)) / 
                    count(img) AS agreement_rate
            ORDER BY predictions DESC
            """).data()
            
            return {
                "total_uploads": total_uploads,
                "agreement_rate": agreement_rate,
                "bin_agreement": bin_agreement
            }
    
    def get_chat_stats(self) -> Dict:
        """Get chat interaction statistics."""
        with self.driver.session() as session:
            total_chats = session.run("""
            MATCH (c:ChatInteraction)
            RETURN count(c) AS total
            """).single()["total"]
            
            # Top waste items mentioned in chats
            top_items = session.run("""
            MATCH (c:ChatInteraction)-[:ABOUT_ITEM]->(w:WasteItem)
            RETURN w.name AS item, count(c) AS mentions
            ORDER BY mentions DESC
            LIMIT 10
            """).data()
            
            # Daily chat counts
            daily_chats = session.run("""
            MATCH (c:ChatInteraction)
            WITH date(c.timestamp) AS day
            RETURN day, count(*) AS chats
            ORDER BY day
            """).data()
            
            return {
                "total_chats": total_chats,
                "top_items": top_items,
                "daily_chats": daily_chats
            }
    
    def get_knowledge_graph_data(self) -> Dict:
        """Get knowledge graph visualization data."""
        with self.driver.session() as session:
            # Get a sample of the knowledge graph with relationships
            graph_data = session.run("""
            MATCH (n)-[r]->(m)
            WHERE 
                (n:User OR n:WasteItem OR n:Bin OR n:SortingAttempt OR n:ChatInteraction OR n:ImageRecognition)
                AND
                (m:User OR m:WasteItem OR m:Bin OR m:SortingAttempt OR m:ChatInteraction OR m:ImageRecognition)
            RETURN n, r, m
            LIMIT 100
            """).data()
            
            # Process for visualization
            nodes = {}
            links = []
            
            for record in graph_data:
                source = record['n']
                target = record['m']
                relationship = record['r']
                
                source_id = source.id
                target_id = target.id
                
                # Add nodes if not already added
                if source_id not in nodes:
                    nodes[source_id] = {
                        "id": source_id,
                        "label": list(source.labels)[0],  # Get the first label
                        "properties": dict(source)
                    }
                
                if target_id not in nodes:
                    nodes[target_id] = {
                        "id": target_id,
                        "label": list(target.labels)[0],
                        "properties": dict(target)
                    }
                
                # Add link
                links.append({
                    "source": source_id,
                    "target": target_id,
                    "type": relationship.type,
                    "properties": dict(relationship)
                })
            
            return {
                "nodes": list(nodes.values()),
                "links": links
            }
    
    def get_user_journey(self, username: str) -> Dict:
        """
        Get a user's journey through the system.
        
        Args:
            username: The username to track
        
        Returns:
            User journey data
        """
        with self.driver.session() as session:
            user = self.get_user_by_username(username)
            if not user:
                return {"error": "User not found"}
            
            user_id = user["user_id"]
            
            # Get all sorting attempts
            sorting_attempts = session.run("""
            MATCH (u:User {user_id: $user_id})-[:ATTEMPTED]->(a:SortingAttempt)-[:FOR_ITEM]->(w:WasteItem)
            MATCH (a)-[:SELECTED]->(chosen:Bin)
            MATCH (a)-[:SHOULD_SELECT]->(correct:Bin)
            RETURN 
                w.name AS item,
                chosen.name AS chosen_bin,
                correct.name AS correct_bin,
                a.is_correct AS is_correct,
                a.timestamp AS timestamp
            ORDER BY a.timestamp DESC
            """, user_id=user_id).data()
            
            # Get chat interactions
            chats = session.run("""
            MATCH (u:User {user_id: $user_id})-[:CHATTED]->(c:ChatInteraction)
            OPTIONAL MATCH (c)-[:ABOUT_ITEM]->(w:WasteItem)
            RETURN 
                c.user_message AS user_message,
                c.ai_response AS ai_response,
                w.name AS about_item,
                c.timestamp AS timestamp
            ORDER BY c.timestamp DESC
            """, user_id=user_id).data()
            
            # Get image recognition
            image_recognitions = session.run("""
            MATCH (u:User {user_id: $user_id})-[:UPLOADED]->(img:ImageRecognition)
            MATCH (img)-[:AI_PREDICTED]->(ai:Bin)
            MATCH (img)-[:USER_SELECTED]->(user:Bin)
            RETURN 
                img.image_name AS image,
                ai.name AS ai_prediction,
                user.name AS user_choice,
                img.is_agreement AS is_agreement,
                img.timestamp AS timestamp
            ORDER BY img.timestamp DESC
            """, user_id=user_id).data()
            
            # Calculate learning progression
            learning_progress = session.run("""
            MATCH (u:User {user_id: $user_id})-[:ATTEMPTED]->(a:SortingAttempt)
            WITH a.timestamp AS timestamp, a.is_correct AS is_correct
            ORDER BY timestamp
            WITH collect(is_correct) AS results
            UNWIND range(0, size(results) - 1) AS i
            WITH i, results[i] AS result
            RETURN 
                i+1 AS attempt_number,
                result AS is_correct,
                sum(CASE WHEN results[i] THEN 1 ELSE 0 END) / (i+1) AS running_accuracy
            ORDER BY attempt_number
            """, user_id=user_id).data()
            
            return {
                "user_info": user,
                "sorting_attempts": sorting_attempts,
                "chats": chats,
                "image_recognitions": image_recognitions,
                "learning_progress": learning_progress
            }

# Global database manager instance
db_manager = None

def get_db_manager():
    """Get or create the Neo4j database manager singleton."""
    global db_manager
    if db_manager is None:
        try:
            db_manager = Neo4jManager()
            logger.info("Successfully created Neo4j database manager")
        except Exception as e:
            logger.error(f"Failed to create Neo4j database manager: {str(e)}")
            raise
    return db_manager

def init_db():
    """Initialize the database."""
    try:
        manager = get_db_manager()
        manager.init_db()
        logger.info("Successfully initialized Neo4j database")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

def add_new_user(username, nationality):
    """Add a new user."""
    manager = get_db_manager()
    return manager.add_user(username, nationality)

def get_user_by_username(username):
    """Get a user by username."""
    manager = get_db_manager()
    return manager.get_user_by_username(username)

def log_sorting_attempt(user_id, waste_item, chosen_bin, correct_bin, is_correct):
    """Log a sorting attempt."""
    manager = get_db_manager()
    manager.log_sorting_attempt(user_id, waste_item, chosen_bin, correct_bin, is_correct)

def log_chat(user_id, user_message, ai_response):
    """Log a chat interaction."""
    manager = get_db_manager()
    manager.log_chat(user_id, user_message, ai_response)

def log_image_recognition(user_id, image_name, ai_bin, user_bin, is_agreement):
    """Log an image recognition interaction."""
    manager = get_db_manager()
    manager.log_image_recognition(user_id, image_name, ai_bin, user_bin, is_agreement) 