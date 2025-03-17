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

from database_manager import (
    init_db, add_new_user, get_user_by_username,
    log_sorting_attempt, log_chat, log_image_recognition
)

# === Configuration ===
app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)
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
        "Nailed it! You’re a recycling rockstar—keep rocking that bin!",
        "Perfect! Even the trash cans are clapping for you!",
        "Spot on! You’re sorting like a pro—Mother Earth says thanks!",
        "Boom! You just made the planet a little happier!",
    ],
    'incorrect': [
        "Oops! That bin’s crying now—better luck next time!",
        "Yikes! That’s the wrong bin—don’t make the trash jealous!",
        "Uh-oh! You just confused the recycling gods!",
        "Not quite! That bin’s like, ‘I’m not your type!’",
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
        
        user_record = get_user_by_username(username)
        if user_record:
            session['user_id'] = user_record[0]
            session['username'] = user_record[1]
        else:
            user_id = add_new_user(username, nationality)
            session['user_id'] = user_id
            session['username'] = username
        
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/')
def index():
    """Main page with login check"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

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
            image_url = url_for('static', filename=image_path.replace('static/', ''))
        else:
            image_url = ''
        
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
    """Check user's sorting answer"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        UserPerformance.init_session_data()
        data = request.json
        item = data.get('item', '').lower().replace(' ', '_')
        selected_container = data.get('container', '')
        
        correct_container = waste_map.get(item, {}).get('container', '')
        is_correct = selected_container == correct_container
        
        log_sorting_attempt(session['user_id'], item, selected_container, 
                          correct_container, is_correct)
        
        session['user_data']['total_attempts'] += 1
        if is_correct:
            session['user_data']['correct_answers'] += 1
        
        item_perf = session['user_data']['item_performance'].setdefault(item, {'attempts': 0, 'correct': 0})
        item_perf['attempts'] += 1
        if is_correct:
            item_perf['correct'] += 1
        
        session.modified = True
        
        feedback = 'Correct!' if is_correct else f'Not quite. This item goes in the {correct_container} bin.'
        tip = random.choice(FUNNY_TIPS['correct' if is_correct else 'incorrect'])
        return jsonify({
            'correct': is_correct,
            'feedback': feedback,
            'correctContainer': correct_container,
            'tip': tip
        })
    except Exception as e:
        logger.error(f"Error checking answer: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_performance')
def get_performance():
    """Return user performance statistics"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    UserPerformance.init_session_data()
    return jsonify({
        'success': True,
        'correct_answers': session['user_data']['correct_answers'],
        'total_attempts': session['user_data']['total_attempts'],
        'item_performance': session['user_data']['item_performance']
    })

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
    message_lower = user_message.lower()
    if 'hello' in message_lower or 'hi' in message_lower:
        return 'Hello! I\'m your waste sorting assistant. How can I help you today?'
    
    waste_items = [item.lower() for item in waste_data['Waste_Item'].tolist()]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([item.replace('_', ' ') for item in waste_items])
    user_vec = vectorizer.transform([user_message])
    
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    if max(similarities) > 0.3:
        idx = similarities.argmax()
        item = waste_items[idx]
        info = waste_map.get(item)
        return f"{item.replace('_', ' ')} goes in the {info['color']} {info['container']} bin."
    
    return "I'm here to help with waste sorting. What would you like to know?"

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
        
        context = ConversationContext()
        response = generate_response(user_message, context)
        log_chat(session['user_id'], user_message, response)
        context.add_message('user', user_message)
        context.add_message('ai', response)
        
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# === Image Recognition ===
def load_resnet_model() -> bool:
    """Load the ResNet model"""
    global resnet_model
    if resnet_model is None:
        try:
            model_path = os.path.join('models', 'resnet.pth')
            if not os.path.exists(model_path):
                logger.error(f"Model file {model_path} not found")
                return False
            
            resnet_model = models.resnet18(weights=None)
            resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, len(class_names))
            resnet_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            resnet_model.eval()
            return True
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            return False
    return True

def preprocess_image(image_bytes: bytes) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    logger.info(f"Processing image bytes: {len(image_bytes)}")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        logger.info(f"Image opened successfully, size: {image.size}")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(image).unsqueeze(0)
        logger.info(f"Tensor shape: {tensor.shape}")
        return tensor, None
    except UnidentifiedImageError as e:
        logger.error(f"Invalid image format: {e}")
        return None, "Invalid image format"
    except Exception as e:
        logger.error(f"Preprocess error: {e}")
        return None, str(e)

def predict_container(image_tensor: torch.Tensor) -> Tuple[Optional[str], float]:
    if not load_resnet_model():
        logger.error("Failed to load ResNet model")
        return None, 0.0
    
    with torch.no_grad():
        outputs = resnet_model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probs, 0)
        logger.info(f"Prediction: {class_names[predicted.item()]}, confidence: {confidence.item() * 100}%")
        return class_names[predicted.item()], confidence.item() * 100

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Handle image upload and classification"""
    if 'user_id' not in session:
        logger.error("Not logged in")
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        if 'waste_image' not in request.files:
            logger.error("No waste_image in request.files")
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['waste_image']
        logger.info(f"Received file: {file.filename}, size: {len(file.read())} bytes")
        file.seek(0)  # Reset file pointer
        image_bytes = file.read()
        logger.info(f"Image bytes length: {len(image_bytes)}")
        image_tensor, error = preprocess_image(image_bytes)
        
        if error:
            logger.error(f"Image preprocessing error: {error}")
            return jsonify({'error': error}), 400
        
        predicted_class, confidence = predict_container(image_tensor)
        if not predicted_class:
            logger.error("Prediction failed, no predicted class")
            return jsonify({'error': 'Prediction failed'}), 500
        
        logger.info(f"Predicted class: {predicted_class}, confidence: {confidence:.2f}%")
        container_details = next(
            (dict(row) for _, row in waste_data.iterrows() 
             if row['Waste_Containers_Type'] == predicted_class), {}
        )
        
        session['last_confidence'] = confidence
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        logger.info(f"Encoded image length: {len(encoded_image)}")
        
        response = {
            'success': True,
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'container_details': container_details,
            'image': encoded_image
        }
        logger.info(f"Response: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
    
    log_image_recognition(session['user_id'], image_path, predicted, user_choice, agreement)
    
    feedback = (
        f"Great! You and the AI agree on {predicted}." if agreement else
        f"You chose {user_choice} while AI predicted {predicted}. Let's learn more!"
    )
    
    return jsonify({'feedback': feedback, 'agreement': agreement})

# === Main Execution ===
if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5001, use_reloader=False)