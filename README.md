# Smart Trash AI

An interactive web application that educates users on proper waste sorting through gamification and AI-powered conversations.

## Features

- **Interactive Waste Sorting Game**: Test your knowledge by sorting different waste items into the correct bins.
- **AI Chat Assistant**: Ask questions about waste sorting and get intelligent responses using natural language processing.
- **Image Recognition**: Upload images of waste items to get AI-powered sorting suggestions using a pre-trained ResNet model.
- **Progress Tracking**: Track your learning progress and identify areas that need improvement.
- **Analytics Dashboard**: Visualize user interactions, performance metrics, and system statistics through an interactive Streamlit dashboard.
- **Knowledge Graph**: View relationships between users, waste items, and interactions in a powerful Neo4j graph database.
- **Data Collection**: All user interactions are logged for analysis and improving the system.

## Technology Stack

- **Backend**: Python with Flask
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: 
  - scikit-learn for natural language processing and similarity matching
  - PyTorch with a pre-trained ResNet model for image recognition
- **Database**: Neo4j AuraDB for graph-based data storage
- **Analytics**: Streamlit and Plotly for the interactive dashboard
- **Deployment**: Docker support for easy deployment

## Installation & Setup

### Prerequisites

1. **Neo4j AuraDB Account**:
   - Sign up for a free [Neo4j AuraDB account](https://neo4j.com/cloud/aura/)
   - Create a new database instance (free tier is sufficient)
   - Keep your connection details (URI, username, password) handy

2. **Copy Environment Variables**:
   - Copy `.env.example` to `.env`
   - Fill in your Neo4j AuraDB credentials in the `.env` file

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

4. Initialize the Neo4j database:
   
   ```
   python setup_neo4j.py
   ```

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

The application uses Neo4j AuraDB to store user interactions as a knowledge graph with the following node types:

- **User**: User information including nationality
- **WasteItem**: Waste items from the classification dataset
- **Bin**: Different types of waste bins (e.g., BioWaste, PaperWaste)
- **SortingAttempt**: Records of waste sorting attempts
- **ChatInteraction**: Records of conversations with the AI assistant
- **ImageRecognition**: Records of image uploads and AI predictions

Relationships between nodes represent various interactions, creating a rich knowledge graph that can be analyzed and visualized.

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
   - **Chat Analytics**: Insights into user questions and topics
   - **Knowledge Graph**: Interactive visualization of the data relationships
   - **User Journey**: Detailed analysis of individual user learning paths

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

## Migration from SQLite

If you're upgrading from the SQLite version:

1. Ensure your SQLite database file (`app_data.db`) is in the project directory
2. Run the migration script:
   ```
   python setup_neo4j.py
   ```
3. Follow the prompts to migrate your existing data to Neo4j

## Extending the Dataset

The application uses a CSV file (`waste_classification.csv`) as its knowledge base. You can extend this by adding more rows to the file with the following format:

```
Waste_Containers_Type,Bin_Color,Category,Waste_Item,image_path
```

For image recognition, place new waste item images in the `static/waste_images/` directory and update the CSV with the relative path.

## Project Structure

- **app.py**: Main Flask application with routes and core functionality
- **neo4j_manager.py**: Neo4j database interface for knowledge graph storage
- **dashboard_app.py**: Streamlit analytics dashboard
- **setup_neo4j.py**: Setup and migration script for Neo4j
- **templates/**: HTML templates for the web interface
- **static/**: CSS, JavaScript, and image assets
- **models/**: Pre-trained machine learning models
- **uploads/**: Temporary storage for user-uploaded images

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
