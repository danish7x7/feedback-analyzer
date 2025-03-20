# Event Feedback Analyzer 📊

A Streamlit web application that analyzes feedback from Google Forms and provides interactive insights using natural language processing and statistical analysis.

## 🌟 Features

- Upload CSV files exported from Google Forms
- Automatic text summarization for open-ended questions
- Statistical analysis for numerical responses
- Common theme extraction
- Interactive chat interface to ask questions about the feedback
- Beautiful visualization of results
- Support for both categorical and numerical data

## 🚀 Live Demo

Access the live application at: [https://feedback_analyzer.streamlit.app](https://danish7x7-feedback-analyzer-streamlit-app-uhgxe8.streamlit.app/)

## 📋 Usage

1. Export your Google Form responses:
   - Open your Google Form
   - Go to "Responses" tab
   - Click the three dots menu (⋮)
   - Select "Download responses (.csv)"

2. Use the application:
   - Upload your CSV file
   - View automatic analysis and summaries
   - Use the chat interface to ask questions about the feedback

Example questions you can ask:
- What are the main themes in the feedback?
- How many responses did we get?
- What did people say about [specific topic]?
- Give me a summary of all feedback

## 💻 Local Development

1. Clone the repository:
```bash
git clone https://github.com/danish7x7/feedback-analyzer.git
cd event-feedback-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

4. Open your browser and go to `http://localhost:8501`

## 🛠️ Technologies Used

- Streamlit for web interface
- NLTK for natural language processing
- Pandas for data processing
- Scikit-learn for text analysis
- Python 3.7+

## 📊 Features in Detail

### Text Analysis
- Automatic summarization of text responses
- Common theme extraction
- Keyword frequency analysis

### Statistical Analysis
- Mean, median, min, and max for numerical responses
- Distribution analysis for categorical responses
- Response count tracking

### Interactive Features
- Expandable sections for each question
- Real-time chat interface for queries
- Dynamic visualization of results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 
