# Event Feedback Analyzer 📊

A powerful, AI-powered web application that transforms Google Forms feedback data into actionable insights. Built with Streamlit and advanced NLP capabilities, this tool provides automated analysis, summaries, and interactive exploration of event feedback data.

## 🌟 Live Demo
Visit the application at: [https://feedback-analyzer.streamlit.app](https://danish7x7-feedback-analyzer-streamlit-app-uhgxe8.streamlit.app/)

## ✨ Features

### 🔄 Automated Analysis
- Instant processing of Google Forms CSV exports
- Smart detection of question types (text, numerical, categorical)
- Advanced text summarization using LexRank algorithm
- Theme extraction using both TF-IDF and frequency analysis
- Statistical analysis for numerical responses

### 💡 Smart Insights
- AI-powered text summarization
- Key sentence extraction
- Theme identification and clustering
- Trend analysis and pattern recognition
- Statistical computations and visualizations

### 🤖 Interactive Chat Interface
- Natural language query processing
- Context-aware responses
- Support for various question types:
  - Summary requests
  - Theme analysis
  - Statistical inquiries
  - Specific topic exploration

### 📊 Visual Analytics
- Clean, modern UI with expandable sections
- Theme visualization with interactive tags
- Statistical metrics with clear visualizations
- Comprehensive data exploration interface

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/danish7x7/feedback-analyzer.git
cd feedback-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## 📋 Usage

1. **Export Your Feedback Data**
   - Open your Google Forms responses
   - Click on the three dots menu (⋮)
   - Select "Download responses (CSV)"

2. **Upload and Analyze**
   - Open the Event Feedback Analyzer
   - Drop your CSV file in the upload area
   - Wait for automatic analysis to complete

3. **Explore Insights**
   - Browse through expandable sections for each question
   - View summaries, themes, and statistics
   - Use the chat interface to ask specific questions

4. **Chat Interface Commands**
   Example questions you can ask:
   - "What are the main themes in the feedback?"
   - "Summarize the responses for [specific question]"
   - "How many people responded?"
   - "What did people say about [topic]?"

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data processing and analysis
- **Sumy**: Advanced text summarization
- **scikit-learn**: Text analysis and TF-IDF
- **NumPy**: Numerical computations
- **Custom NLP**: Text processing and theme extraction

## 🔒 Privacy & Security

- All processing is done locally
- No data is stored or transmitted
- Secure file handling
- No external API dependencies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit for the amazing web app framework
- Sumy for text summarization capabilities
- The open-source community for various tools and libraries

## 📬 Contact

Danish - [danishbirsingh.bhatti@sjsu.edu]

Project Link: [https://github.com/danish7x7/feedback-analyzer](https://github.com/danish7x7/feedback-analyzer)

---
Made with ❤️ for event organizers and feedback analysts 
