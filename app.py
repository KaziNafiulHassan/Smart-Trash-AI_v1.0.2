# app.py
import os
import json
import random
import io
import base64
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging

from flask import (
    Flask, render_template, request, jsonify, session, 
    redirect, url_for, make_response
)
from werkzeug.utils import secure_filename
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, UnidentifiedImageError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np  # Required for cosine_similarity

from neo4j_manager import (
    init_db, add_new_user, get_user_by_username,
    log_sorting_attempt, log_chat, log_image_recognition, get_db_manager
)

# === Configuration ===
app = Flask(__name__, static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Debugging: Log static folder path
print(f"Static folder path: {os.path.abspath(app.static_folder)}")

# === Global Variables ===
waste_data = pd.read_csv('waste_classification.csv')
waste_map = {row['Waste_Item'].lower(): {
    'container': row['Waste_Containers_Type'],
    'color': row['Bin_Color'],
    'category': row['Category']
} for _, row in waste_data.iterrows()}

class_names = ['BioWaste', 'Depot_Container_Glass', 'LightweightPackaging_LVP', 
              'PaperWaste', 'ResidualWaste']
resnet_model = None

FUNNY_TIPS = {
    'correct': [
        "Nailed it! You're a recycling rockstar—keep rocking that bin!",
        "Perfect! Even the trash cans are clapping for you!",
        "Spot on! You're sorting like a pro—Mother Earth says thanks!",
        "Boom! You just made the planet a little happier!",
    ],
    'incorrect': [
        "Oops! That bin's crying now—better luck next time!",
        "Yikes! That's the wrong bin—don't make the trash jealous!",
        "Uh-oh! You just confused the recycling gods!",
        "Not quite! That bin's like, 'I'm not your type!'",
    ]
}

# === Utility Functions ===
def add_no_cache_headers(response):
    """Add no-cache headers to response"""
    if 'text/html' in response.headers.get('Content-Type', ''):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# === Authentication ===
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login and registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        nationality = request.form.get('nationality')
        
        if not username or not nationality:
            return jsonify({'error': 'Username and nationality required'}), 400
        
        try:
            user_record = get_user_by_username(username)
            if user_record:
                session['user_id'] = user_record['user_id']
                session['username'] = user_record['username']
            else:
                user_id = add_new_user(username, nationality)
                session['user_id'] = user_id
                session['username'] = username
        except Exception as e:
            logger.error(f"Database error during login: {str(e)}")
            return jsonify({'error': 'Database connection error. Please try again later.'}), 500
        
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/')
def index():
    """Main page with login check"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username', 'Guest'))

@app.route('/db_status')
def db_status():
    """Check database connection status"""
    try:
        db_manager = get_db_manager()
        # Test connection with a simple query
        with db_manager.driver.session() as session:
            result = session.run("RETURN 'Connected!' AS message")
            message = result.single()["message"]
        return jsonify({
            'status': 'connected',
            'message': message
        })
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# === Sorting Game ===
class UserPerformance:
    """Manage user performance tracking"""
    @staticmethod
    def init_session_data():
        if 'user_data' not in session:
            session['user_data'] = {
                'correct_answers': 0,
                'total_attempts': 0,
                'item_performance': {}
            }

@app.route('/get_random_item')
def get_random_item():
    """Get a random waste item for the sorting game"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        random_item = random.choice(waste_data['Waste_Item'].tolist())
        formatted_item = random_item.replace('_', ' ')
        container_info = waste_map.get(random_item.lower(), {})
        
        # Debug: Log all columns to ensure image_path exists
        logger.info(f"Available columns in waste_data: {list(waste_data.columns)}")
        
        # Get the image path with case-insensitive matching
        matching_rows = waste_data[waste_data['Waste_Item'].str.lower() == random_item.lower()]
        if not matching_rows.empty and 'image_path' in waste_data.columns:
            image_path = matching_rows['image_path'].iloc[0]
        else:
            image_path = ''
        
        logger.info(f"Item: {random_item}, Image Path: {image_path}")
        if not image_path:
            logger.warning(f"No image path found for item: {random_item}")
        
        # Construct the correct URL for the image path
        if image_path:
            # Make sure we're using the correct path format
            if image_path.startswith('static/'):
                image_path = image_path.replace('static/', '')
            # Ensure we're pointing to the waste_images folder
            if not image_path.startswith('waste_images/'):
                image_path = f'waste_images/{os.path.basename(image_path)}'
            image_url = url_for('static', filename=image_path)
        else:
            image_url = ''
        
        logger.info(f"Final image URL: {image_url}")
        
        return jsonify({
            'item': formatted_item,
            'container': container_info.get('container', ''),
            'color': container_info.get('color', ''),
            'category': container_info.get('category', '').replace('_', ' '),
            'image_path': image_url
        })
    except Exception as e:
        logger.error(f"Error in get_random_item: {str(e)}")
        return jsonify({'error': 'Failed to get random item'}), 500

@app.route('/check_answer', methods=['POST'])
def check_answer():
    """Check if the selected container is correct for the waste item"""
    # Log the request for debugging
    logger.info(f"Received check_answer request: {request.json}")
    
    # Temporarily disable session check for debugging
    # if 'user_id' not in session:
    #     return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        if not data:
            logger.error("No JSON data received in request")
            return jsonify({'error': 'No data received'}), 400
            
        item = data.get('item')
        selected_container = data.get('container')
        
        logger.info(f"Checking answer for item: {item}, container: {selected_container}")
        
        if not item or not selected_container:
            return jsonify({'error': 'Missing item or container'}), 400
        
        # Find the correct container for this item
        correct_container = None
        item_lower = item.lower().strip()
        
        # First try exact match
        for _, row in waste_data.iterrows():
            if row['Waste_Item'].lower().strip() == item_lower:
                correct_container = row['Waste_Containers_Type']
                break
        
        # If no exact match, try partial match
        if not correct_container:
            for _, row in waste_data.iterrows():
                if item_lower in row['Waste_Item'].lower().strip() or row['Waste_Item'].lower().strip() in item_lower:
                    correct_container = row['Waste_Containers_Type']
                    logger.info(f"Found partial match: '{item}' matches with '{row['Waste_Item']}'")
                    break
        
        # If still no match, use a default container based on common items
        if not correct_container:
            # Map common waste types to containers
            common_waste_mapping = {
                'paper': 'PaperWaste',
                'cardboard': 'PaperWaste',
                'plastic': 'LightweightPackaging_LVP',
                'bottle': 'Depot_Container_Glass',
                'glass': 'Depot_Container_Glass',
                'food': 'BioWaste',
                'organic': 'BioWaste',
                'metal': 'LightweightPackaging_LVP',
                'electronic': 'ElectronicWaste',
                'battery': 'HazardousWaste',
                'hygiene': 'ResidualWaste',
                'diaper': 'ResidualWaste',
                'toilet': 'ResidualWaste',
                'waste': 'ResidualWaste',
                'cooked': 'BioWaste'
            }
            
            for keyword, container in common_waste_mapping.items():
                if keyword in item_lower:
                    correct_container = container
                    logger.info(f"Using keyword mapping for '{item}': keyword '{keyword}' maps to '{container}'")
                    break
        
        # If still no match, default to ResidualWaste
        if not correct_container:
            correct_container = 'ResidualWaste'  # Default container
            logger.warning(f"No match found for '{item}', using default container: {correct_container}")
        
        # Check if the answer is correct
        is_correct = selected_container == correct_container
        
        # Get feedback and tip
        feedback = f"{'Correct!' if is_correct else 'Incorrect.'} {item} belongs in {correct_container}."
        tip = random.choice(FUNNY_TIPS['correct' if is_correct else 'incorrect'])
        
        # Update session statistics if session exists
        if 'user_id' in session:
            correct_answers = session.get('correct_answers', 0)
            total_attempts = session.get('total_attempts', 0)
            
            if is_correct:
                session['correct_answers'] = correct_answers + 1
            session['total_attempts'] = total_attempts + 1
            
            # Track bin usage for charts
            bin_count_key = f'bin_count_{selected_container}'
            session[bin_count_key] = session.get(bin_count_key, 0) + 1
            
            # Log attempt to database if available
            user_id = session.get('user_id')
            if user_id:
                try:
                    log_sorting_attempt(
                        user_id=user_id,
                        item=item,
                        correct_container=correct_container,
                        selected_container=selected_container,
                        is_correct=is_correct
                    )
                except Exception as e:
                    logger.error(f"Failed to log sorting attempt: {str(e)}")
        
        logger.info(f"Answer check result: correct={is_correct}, feedback={feedback}")
        
        return jsonify({
            'correct': is_correct,
            'item': item,
            'container': selected_container,
            'correct_container': correct_container,
            'feedback': feedback,
            'tip': tip
        })
    except Exception as e:
        logger.error(f"Error checking answer: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_performance', methods=['GET'])
def get_performance():
    """Get user's performance data for the sorting game"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        user_id = session['user_id']
        db_manager = get_db_manager()
        
        # Get basic performance stats
        correct_answers = session.get('correct_answers', 0)
        total_attempts = session.get('total_attempts', 0)
        
        # Get bin statistics from database if available
        bin_stats = {}
        try:
            if db_manager:
                # Query for bin usage statistics
                query = """
                MATCH (u:User {id: $user_id})-[:ATTEMPTED]->(a:SortingAttempt)
                RETURN a.selected_container AS bin, COUNT(a) AS count
                """
                result = db_manager.query(query, {'user_id': user_id})
                
                if result:
                    for record in result:
                        bin_stats[record['bin']] = record['count']
                        
                # If no data in database, use session data
                if not bin_stats:
                    raise Exception("No bin stats in database")
        except Exception as e:
            logger.warning(f"Could not get bin stats from database: {str(e)}")
            # Fallback to session data or create mock data
            bin_types = ['BioWaste', 'PaperWaste', 'LightweightPackaging_LVP', 'Depot_Container_Glass', 'ResidualWaste']
            for bin_type in bin_types:
                bin_stats[bin_type] = session.get(f'bin_count_{bin_type}', 0)
        
        return jsonify({
            'success': True,
            'correct_answers': correct_answers,
            'total_attempts': total_attempts,
            'bin_stats': bin_stats
        })
    except Exception as e:
        logger.error(f"Error getting performance: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# === Chat System ===
class ConversationContext:
    """Manage conversation history"""
    def __init__(self, max_messages=10):
        self.messages = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str):
        self.messages.append({'role': role, 'content': content, 'timestamp': datetime.now().isoformat()})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self):
        return self.messages

def generate_response(user_message: str, context: ConversationContext) -> str:
    """Generate AI response based on user message"""
    try:
        message_lower = user_message.lower()
        if 'hello' in message_lower or 'hi' in message_lower:
            return 'Hello! I\'m your waste sorting assistant. How can I help you today?'
        
        # Get waste items from the dataset
        waste_items = [item.lower() for item in waste_data['Waste_Item'].tolist()]
        
        # Create TF-IDF vectorizer
        try:
            vectorizer = TfidfVectorizer()
            item_texts = [item.replace('_', ' ') for item in waste_items]
            
            # Handle empty input
            if not item_texts:
                return "I'm having trouble accessing the waste data. Please try again later."
            
            tfidf_matrix = vectorizer.fit_transform(item_texts)
            user_vec = vectorizer.transform([user_message])
            
            # Calculate similarity
            similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
            
            # If there's a good match
            if max(similarities) > 0.3:
                idx = similarities.argmax()
                item = waste_items[idx]
                info = waste_map.get(item)
                
                if info:
                    return f"{item.replace('_', ' ')} goes in the {info['color']} {info['container']} bin."
                else:
                    return f"I recognize {item.replace('_', ' ')} but don't have disposal information for it."
        except Exception as e:
            logger.error(f"Error in text processing: {str(e)}")
            return "I'm having trouble processing your request. Could you try rephrasing?"
        
        return "I'm here to help with waste sorting. You can ask me which bin a specific item goes in."
    except Exception as e:
        logger.error(f"General error in generate_response: {str(e)}")
        return "Sorry, I encountered an error. Please try again."

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat interactions"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({'response': 'No message provided'}), 400
        
        logger.info(f"Chat request received: {user_message}")
        
        context = ConversationContext()
        response = generate_response(user_message, context)
        
        try:
            # Log chat to Neo4j
            log_chat(session['user_id'], user_message, response)
        except Exception as e:
            logger.error(f"Failed to log chat: {str(e)}")
            # Continue even if logging fails
        
        context.add_message('user', user_message)
        context.add_message('ai', response)
        
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Sorry, I encountered an error processing your request.'}), 500

# === Image Recognition ===
def cleanup_memory():
    """Force garbage collection and clear memory caches"""
    import gc
    import torch
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    logger.info("Memory cleanup performed")

def load_resnet_model():
    """Load the ResNet model with fallback to pre-trained if fine-tuned model is missing"""
    global resnet_model
    
    if resnet_model is not None:
        return True
    
    # Clean up memory before loading model
    cleanup_memory()
    
    # Check if the fine-tuned model file exists
    model_path = os.path.join('models', 'resnet.pth')
    use_pretrained_fallback = not os.path.exists(model_path)
    
    if use_pretrained_fallback:
        logger.warning(f"Fine-tuned model file {model_path} not found, using pre-trained ResNet18 as fallback")
    else:
        logger.info(f"Loading fine-tuned ResNet model from {model_path}...")
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Load ResNet18 model
            if use_pretrained_fallback:
                # Use pre-trained weights if no fine-tuned model is available
                model = models.resnet18(weights='IMAGENET1K_V1')
            else:
                # Load without pre-trained weights, we'll load our fine-tuned weights
                model = models.resnet18(weights=None)
            
            # Modify the final layer for our 5 waste categories
            model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
            
            # Load the fine-tuned weights if available
            if not use_pretrained_fallback:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                logger.info("Fine-tuned weights loaded successfully")
            
            # Set model to evaluation mode
            model.eval()
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            resnet_model = model
            logger.info("ResNet model loaded successfully")
            return True
        except Exception as e:
            retry_count += 1
            logger.error(f"Error loading ResNet model (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count >= max_retries:
                logger.error("Failed to load ResNet model after multiple attempts")
                return False
            # Wait before retrying
            import time
            time.sleep(2)
    
    return False

def preprocess_image(image_bytes: bytes):
    """Preprocess image for model prediction with memory optimization"""
    try:
        # Try to open the image with PIL
        try:
            # Use a smaller initial size to reduce memory usage
            image = Image.open(io.BytesIO(image_bytes))
            
            # Immediately resize to a smaller size to reduce memory usage
            if max(image.size) > 400:
                image.thumbnail((400, 400), Image.LANCZOS)
                logger.info(f"Resized image to {image.size} to save memory")
        except UnidentifiedImageError:
            logger.error("Cannot identify image file format")
            return None, "Cannot identify image file format. Please upload a valid image file."
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            return None, f"Error opening image: {str(e)}"
        
        # Convert to RGB if needed (handles PNG with transparency, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image from {image.mode} to RGB")
        
        # Use a simpler transformation pipeline to reduce memory usage
        preprocess = transforms.Compose([
            transforms.Resize(224),  # Single resize instead of resize + center crop
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Apply transformations
        image_tensor = preprocess(image).unsqueeze(0)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return image_tensor, None
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None, f"Error preprocessing image: {str(e)}"

def predict_container(image_tensor: torch.Tensor) -> Tuple[Optional[str], float]:
    if not load_resnet_model():
        logger.error("Failed to load model")
        return None, 0.0
    
    try:
        # Use torch.no_grad to reduce memory usage during inference
        with torch.no_grad():
            # Run the prediction with a timeout using threading (works on Windows)
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def prediction_worker():
                try:
                    # Run the prediction
                    outputs = resnet_model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Get the confidence scores using softmax
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    confidence = probabilities[predicted[0]].item() * 100
                    
                    # Get the predicted class name
                    predicted_class = class_names[predicted[0]]
                    
                    # Put the result in the queue
                    result_queue.put((predicted_class, confidence))
                except Exception as e:
                    logger.error(f"Error in prediction worker: {str(e)}")
                    result_queue.put((None, 0.0))
            
            # Start the prediction in a separate thread
            prediction_thread = threading.Thread(target=prediction_worker)
            prediction_thread.daemon = True
            prediction_thread.start()
            
            # Wait for the result with a timeout
            try:
                # 30-second timeout
                result = result_queue.get(timeout=30)
                return result
            except queue.Empty:
                logger.error("Prediction timed out after 30 seconds")
                return None, 0.0
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return None, 0.0

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Handle image upload and classification with memory optimization"""
    if 'user_id' not in session:
        logger.error("Not logged in")
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        # Check if the request has the file part
        if 'waste_image' not in request.files:
            logger.error("No waste_image in request.files")
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['waste_image']
        
        # Check if the file is empty
        if file.filename == '':
            logger.error("Empty file submitted")
            return jsonify({'error': 'Empty file submitted'}), 400
        
        # Check file size before reading it completely
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        
        # Log file details
        logger.info(f"Received file: {file.filename}, size: {file_size} bytes")
        
        # Check if file is too large (5MB limit - reduced from 10MB)
        if file_size > 5 * 1024 * 1024:  # 5MB
            logger.error(f"File too large: {file_size} bytes")
            return jsonify({'error': 'File too large. Maximum size is 5MB'}), 413
        
        # Read the file in one go for small files
        image_bytes = file.read()
        logger.info(f"Image bytes length: {len(image_bytes)}")
        
        # Process the image
        try:
            # Preprocess image (this now includes resizing to save memory)
            image_tensor, error = preprocess_image(image_bytes)
            
            if error:
                logger.error(f"Image preprocessing error: {error}")
                return jsonify({'error': error}), 400
            
            # Force garbage collection before prediction
            import gc
            gc.collect()
            
            # Predict waste container
            predicted_class, confidence = predict_container(image_tensor)
            if not predicted_class:
                logger.error("Prediction failed, no predicted class")
                return jsonify({'error': 'Prediction failed'}), 500
            
            logger.info(f"Predicted class: {predicted_class}, confidence: {confidence:.2f}%")
            
            # Find container details - simplified to reduce memory usage
            container_details = {}
            for _, row in waste_data.iterrows():
                if row['Waste_Containers_Type'] == predicted_class:
                    # Only include essential fields
                    container_details = {
                        'Waste_Containers_Type': row['Waste_Containers_Type'],
                        'Bin_Color': row['Bin_Color'],
                        'Category': row['Category']
                    }
                    break
            
            # Store in session
            session['last_confidence'] = confidence
            session['last_image_path'] = file.filename if file.filename else 'uploaded_image'
            session['last_predicted_bin'] = predicted_class
            
            # Compress the image before encoding to reduce size
            try:
                img = Image.open(io.BytesIO(image_bytes))
                # Resize to a smaller size to reduce memory usage
                img.thumbnail((300, 300))
                # Compress the image with higher compression
                output_buffer = io.BytesIO()
                img.save(output_buffer, format='JPEG', quality=50)
                compressed_bytes = output_buffer.getvalue()
                encoded_image = base64.b64encode(compressed_bytes).decode('utf-8')
                
                # Clear variables to free memory
                output_buffer = None
                compressed_bytes = None
                img = None
                gc.collect()
            except Exception as img_err:
                logger.error(f"Image compression error: {str(img_err)}")
                # Don't return the image if compression fails
                encoded_image = ""
            
            logger.info(f"Encoded image length: {len(encoded_image)}")
            
            response = {
                'success': True,
                'prediction': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'container_details': container_details,
                'image': encoded_image
            }
            
            # Force garbage collection before returning
            gc.collect()
            
            return jsonify(response)
        except Exception as process_err:
            logger.error(f"Image processing error: {str(process_err)}")
            return jsonify({'error': f"Image processing error: {str(process_err)}"}), 500
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/verify_prediction', methods=['POST'])
def verify_prediction():
    """Verify user's feedback on image prediction"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    predicted = data.get('predicted_container', '')
    user_choice = data.get('user_container', '')
    
    agreement = predicted == user_choice
    image_path = session.get('last_image_path', 'unknown')
    
    try:
        log_image_recognition(session['user_id'], image_path, predicted, user_choice, agreement)
    except Exception as e:
        logger.error(f"Error logging image recognition: {str(e)}")
        # Continue even if logging fails
    
    feedback = (
        f"Great! You and the AI agree on {predicted}." if agreement else
        f"You chose {user_choice} while AI predicted {predicted}. Let's learn more!"
    )
    
    return jsonify({'feedback': feedback, 'agreement': agreement})

# === Main Execution ===
if __name__ == '__main__':
    try:
        init_db()
        logger.info("Neo4j database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        # Continue running even if database initialization fails
        # This will allow the app to start and show an error message to users
        # rather than failing to start completely
    
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)