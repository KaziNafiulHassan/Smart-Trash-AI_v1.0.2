# Smart Trash AI

An interactive web application that educates users on proper waste sorting through gamification and AI-powered conversations.

## Features

- **Interactive Waste Sorting Game**: Test your knowledge by sorting different waste items into the correct bins.
- **AI Chat Assistant**: Ask questions about waste sorting and get intelligent responses using natural language processing.
- **Image Recognition**: Upload images of waste items to get AI-powered sorting suggestions using a pre-trained ResNet model.
- **Progress Tracking**: Track your learning progress and identify areas that need improvement.
- **Analytics Dashboard**: Visualize user interactions, performance metrics, and system statistics through an interactive Streamlit dashboard.
- **Data Collection**: All user interactions are logged for analysis and improving the system.

## Technology Stack

- **Backend**: Python with Flask
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: 
  - scikit-learn for natural language processing and similarity matching
  - PyTorch with a pre-trained ResNet model for image recognition
- **Database**: SQLite for storing user interactions and feedback
- **Analytics**: Streamlit and Plotly for the interactive dashboard
- **Deployment**: Docker support for easy deployment

## Installation & Setup

### Standard Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/smart-trash-ai.git
   cd smart-trash-ai
   ```

2. Set up a virtual environment (recommended):

   ```
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Initialize the database:
   
   The application uses SQLite and will automatically create the database on first run.

5. Run the application:

   ```
   python app.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:5001
   ```

### Docker Setup

1. Build and run using Docker Compose:

   ```
   docker-compose up --build
   ```

2. For debugging purposes:

   ```
   docker-compose -f docker-compose.debug.yml up --build
   ```

## Database Structure

The application uses SQLite to store user interactions. The database contains:

- **users**: Basic user information including nationality
- **sorting_game_logs**: Records of user waste sorting attempts
- **chat_logs**: Records of conversations with the AI assistant
- **image_recognition_logs**: Records of image uploads and AI predictions

## Analytics Dashboard

The application includes a Streamlit analytics dashboard that provides insights into user interactions and system performance:

1. Run the dashboard:

   ```
   streamlit run dashboard_app.py
   ```

2. The dashboard provides:
   - **User Analytics**: Statistics on user signups and nationality distribution
   - **Sorting Game Analytics**: Performance metrics for different waste items and overall accuracy
   - **Image Recognition Analytics**: Analysis of AI-user agreement rates for different waste bins

## How to Use

### Sorting Game

1. Log in with your username and nationality
2. The application will present you with a waste item
3. Click on the bin where you think the item should go
4. Receive immediate feedback on your choice
5. Click "Next Item" to continue playing

### Chat Assistant

1. Navigate to the Chat tab
2. Type questions about waste sorting or specific items
3. Get responses from the AI assistant based on the waste classification dataset

### Image Recognition

1. Navigate to the Image Recognition tab
2. Upload an image of a waste item
3. The AI will predict which bin the item belongs in
4. Verify if you agree with the prediction to help improve the system

## Extending the Dataset

The application uses a CSV file (`waste_classification.csv`) as its knowledge base. You can extend this by adding more rows to the file with the following format:

```
Waste_Containers_Type,Bin_Color,Category,Waste_Item,image_path
```

For image recognition, place new waste item images in the `static/waste_images/` directory and update the CSV with the relative path.

## Project Structure

- **app.py**: Main Flask application with routes and core functionality
- **database_manager.py**: SQLite database interface
- **dashboard_app.py**: Streamlit analytics dashboard
- **templates/**: HTML templates for the web interface
- **static/**: CSS, JavaScript, and image assets
- **models/**: Pre-trained machine learning models
- **uploads/**: Temporary storage for user-uploaded images

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
