# Smart Trash AI

An interactive web application that educates users on proper waste sorting through gamification and AI-powered conversations.

## Features

- **Interactive Waste Sorting Game**: Test your knowledge by sorting different waste items into the correct bins.
- **AI Chat Assistant**: Ask questions about waste sorting and get intelligent responses.
- **Image Recognition**: Upload images of waste items to get AI-powered sorting suggestions.
- **Progress Tracking**: Track your learning progress and identify areas that need improvement.
- **Reinforcement Learning**: The system tracks user performance to provide personalized feedback.
- **Data Collection**: All user interactions are logged for analysis and improving the system.

## Technology Stack

- **Backend**: Python with Flask
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: scikit-learn for natural language processing and similarity matching
- **Computer Vision**: PyTorch with a pre-trained ResNet model for image recognition
- **Database**: MongoDB for storing user interactions and feedback
- **Reinforcement Learning**: Multi-armed bandit, Q-learning, and POMDP algorithms for adaptive content

## Installation & Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/smart-trash-ai.git
   cd smart-trash-ai
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up MongoDB:

   - Install MongoDB from [mongodb.com](https://www.mongodb.com/try/download/community)
   - Start the MongoDB service
   - The application will automatically create the required database and collections

4. Run the application:

   ```
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5001
   ```

## Database Structure

The application uses MongoDB to store user interactions. The database contains:

- **quiz_logs**: Records of user quiz interactions
- **sorting_logs**: Records of waste sorting attempts
- **conversation_logs**: Records of conversations with the AI assistant

## Analytics Dashboard

The application includes a standalone analytics dashboard that provides insights into user interactions and system performance:

1. Run the dashboard:

   ```
   python analytics_dashboard.py
   ```

2. The dashboard provides:
   - **Overview**: System statistics including total users, sorting attempts, and success rates
   - **Sorting Analysis**: Performance metrics for different waste items
   - **Chat Analysis**: Analysis of chat interactions and message patterns
   - **User Analysis**: Individual user learning curves and progress tracking

The dashboard automatically connects to your MongoDB database to retrieve and visualize the data.

### Dashboard Requirements

- Tkinter (comes pre-installed with Python)
- Matplotlib for data visualization
- MongoDB connection

### Screenshots

_(Coming soon)_

## How to Use

### Sorting Game

1. The application will present you with a waste item.
2. Click on the bin where you think the item should go.
3. Receive immediate feedback on your choice.
4. Click "Next Item" to continue playing.

### Chat Assistant

1. Navigate to the Chat tab.
2. Type questions about waste sorting or specific items.
3. Get responses from the AI assistant based on the waste classification dataset.

### Image Recognition

1. Navigate to the Image Recognition tab.
2. Upload an image of a waste item.
3. Select which bin you think the item belongs in.
4. See the AI's prediction and get educational feedback.

### Progress Tracking

1. Visit the Progress tab to see your overall performance.
2. Review items that you need more practice with.
3. Track your improvement over time.

## Extending the Dataset

The application uses a CSV file (`waste_classification.csv`) as its knowledge base. You can extend this by adding more rows to the file with the following format:

```
Waste_Containers_Type,Bin_Color,Category,Waste_Item
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
