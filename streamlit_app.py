import streamlit as st

# Set page config at the very beginning
st.set_page_config(
    page_title="Event Feedback Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        color: #1E88E5;
        font-size: 3rem !important;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #424242;
        font-size: 1.2rem !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card-like styling for sections */
    .css-1r6slb0 {  /* Expander class */
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Theme tag styling */
    .theme-tag {
        display: inline-block;
        background: linear-gradient(135deg, #1E88E5 0%, #1976D2 100%);
        color: white;
        padding: 4px 12px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metrics styling */
    .css-1xarl3l {  /* Metric value class */
        font-size: 1.8rem !important;
        color: #1E88E5;
    }
    
    /* Chat input styling */
    .stTextInput input {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: 2px solid #E3F2FD;
    }
    
    .stTextInput input:focus {
        border-color: #1E88E5;
        box-shadow: 0 0 0 2px rgba(30,136,229,0.2);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 20px;
        padding: 0.5rem 2rem;
        background: linear-gradient(135deg, #1E88E5 0%, #1976D2 100%);
        color: white;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* File uploader styling */
    .css-1v0mbdj {  /* File uploader class */
        border-radius: 10px;
        border: 2px dashed #1E88E5;
        padding: 2rem;
        text-align: center;
        background: #F8F9FA;
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict

# Custom tokenization functions that don't rely on NLTK
def custom_sentence_tokenize(text):
    """Split text into sentences without NLTK"""
    # Basic sentence splitting on period, question mark, exclamation mark
    # followed by a space and uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Clean up sentences
    return [s.strip() for s in sentences if s.strip()]

def custom_word_tokenize(text):
    """Split text into words without NLTK"""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace
    return [word for word in text.split() if word]

# Common English stopwords (abbreviated list)
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 
    'won', 'wouldn'
])

class FeedbackAnalyzer:
    def __init__(self, df):
        self.df = df
        self.analysis_results = self.analyze_feedback()
        self.feedback_context = self.create_feedback_context()

    def analyze_feedback(self):
        """
        Analyze feedback data and generate summaries for each question
        """
        summaries = {}
        
        for column in self.df.columns:
            if column.lower() in ['timestamp', 'email', 'name']:  # Skip metadata columns
                continue
            
            responses = self.df[column].dropna().tolist()
            if not responses:
                continue
            
            # Basic statistics
            response_count = len(responses)
            
            # Text analysis
            if isinstance(responses[0], str):
                # For text responses
                summary = self.generate_text_summary(responses)
            else:
                # For numerical/categorical responses
                summary = self.generate_numerical_summary(responses)
            
            summaries[column] = {
                'question': column,
                'response_count': response_count,
                'summary': summary,
                'responses': responses  # Store actual responses for context
            }
        
        return summaries

    def generate_text_summary(self, responses):
        """
        Generate a summary for text responses
        """
        # Combine all responses
        combined_text = ' '.join(responses)
        
        # Tokenize sentences safely using our custom function
        sentences = custom_sentence_tokenize(combined_text)
        
        if not sentences:
            return {
                'text_summary': "No text responses available.",
                'common_themes': [],
                'total_responses': len(responses)
            }
        
        # Calculate sentence importance using TF-IDF
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Get average TF-IDF scores for each sentence
            importance_scores = tfidf_matrix.sum(axis=1).A1
            
            # Get top sentences (most representative)
            if len(sentences) >= 3:
                top_sentence_indices = importance_scores.argsort()[-3:][::-1]
                summary_sentences = [sentences[i] for i in top_sentence_indices]
            else:
                summary_sentences = sentences
            
            # Create a summary paragraph
            summary = " ".join(summary_sentences)
        except Exception:
            # Fallback to a simple summary if TF-IDF fails
            summary = ". ".join(sentences[:min(3, len(sentences))])
        
        # Add some basic statistics
        common_themes = self.extract_common_themes(responses)
        
        return {
            'text_summary': summary,
            'common_themes': common_themes,
            'total_responses': len(responses)
        }

    def generate_numerical_summary(self, responses):
        """
        Generate a summary for numerical or categorical responses
        """
        series = pd.Series(responses)
        
        if series.dtype in ['int64', 'float64']:
            return {
                'mean': series.mean(),
                'median': series.median(),
                'min': series.min(),
                'max': series.max()
            }
        else:
            # For categorical data
            value_counts = series.value_counts()
            return {
                'most_common': value_counts.index[0],
                'distribution': value_counts.to_dict()
            }

    def extract_common_themes(self, responses):
        """
        Extract common themes from text responses
        """
        try:
            # Tokenize using our custom function
            all_words = []
            for response in responses:
                words = custom_word_tokenize(response)
                all_words.extend(words)
            
            # Filter stopwords and short words
            words = [word for word in all_words if word not in STOPWORDS and len(word) > 3]
            
            # Count word frequencies
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            
            # Get top 5 most common words
            common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            return [{'word': word, 'count': count} for word, count in common_words]
        except Exception:
            # Return empty list if anything fails
            return []

    def create_feedback_context(self):
        """
        Create a comprehensive context string for the chatbot
        """
        context = "Based on the feedback data analysis:\n\n"
        
        for question, data in self.analysis_results.items():
            context += f"Question: {question}\n"
            context += f"Number of responses: {data['response_count']}\n"
            
            summary = data['summary']
            if 'text_summary' in summary:
                context += f"Summary: {summary['text_summary']}\n"
                context += "Common themes: " + ", ".join(f"{theme['word']} ({theme['count']} mentions)" 
                                                       for theme in summary['common_themes']) + "\n"
            elif 'mean' in summary:
                context += f"Statistical summary:\n"
                context += f"- Mean: {summary['mean']}\n"
                context += f"- Median: {summary['median']}\n"
                context += f"- Range: {summary['min']} to {summary['max']}\n"
            else:
                context += f"Most common response: {summary['most_common']}\n"
                context += "Distribution: " + ", ".join(f"{k}: {v}" for k, v in summary['distribution'].items()) + "\n"
            
            context += "\n"
        
        return context

    def answer_question(self, question):
        """
        Answer questions about the feedback data
        """
        if not self.analysis_results:
            return "No feedback data has been analyzed yet."
            
        # Define common question patterns and their handling logic
        if "how many" in question.lower() and "responses" in question.lower():
            try:
                total_responses = max(data['response_count'] for data in self.analysis_results.values())
                return f"There are {total_responses} total responses in the feedback data."
            except Exception:
                return "I couldn't determine the number of responses. Please try again."
        
        if "common themes" in question.lower() or "main themes" in question.lower():
            try:
                themes = []
                for data in self.analysis_results.values():
                    if 'text_summary' in data['summary'] and 'common_themes' in data['summary']:
                        themes.extend(theme['word'] for theme in data['summary']['common_themes'])
                if themes:
                    return f"The main themes identified across all responses are: {', '.join(set(themes))}."
                else:
                    return "No common themes were identified in the feedback."
            except Exception:
                return "I had trouble identifying common themes. Please try another question."
            
        if "summary" in question.lower():
            return self.feedback_context
        
        # For questions about specific questions/topics, search through the data
        try:
            for question_text, data in self.analysis_results.items():
                if any(word in question_text.lower() for word in question.lower().split()):
                    if 'text_summary' in data['summary']:
                        return f"Regarding '{question_text}': {data['summary']['text_summary']}"
                    elif 'mean' in data['summary']:
                        return f"For '{question_text}', the average is {data['summary']['mean']:.2f} (range: {data['summary']['min']} to {data['summary']['max']})"
                    else:
                        return f"For '{question_text}', the most common response was '{data['summary']['most_common']}'"
        except Exception:
            return "I had trouble finding information about that specific topic. Please try another question."
        
        return "I'm not sure about that specific aspect of the feedback. Could you rephrase your question or ask about a specific question from the survey?"

def main():
    # Title with custom styling
    st.markdown('<h1 class="main-title">✨ Event Feedback Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Transform your feedback data into actionable insights with AI-powered analysis</p>', unsafe_allow_html=True)

    # File upload with friendly message and proper accessibility
    st.markdown("### 📁 Upload Your Feedback Data")
    st.markdown("Drop your Google Forms CSV file below to get started!")
    uploaded_file = st.file_uploader(
        label="Upload CSV file",
        type="csv",
        help="Upload a CSV file exported from Google Forms",
        label_visibility="collapsed"  # Hide the label but keep it accessible
    )

    if uploaded_file is not None:
        try:
            # Add a spinner for better UX
            with st.spinner("🔍 Analyzing your feedback data..."):
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                if df.empty:
                    st.error("📭 The uploaded file appears to be empty. Please check your CSV file and try again.")
                    return
                    
                # Create analyzer instance
                analyzer = FeedbackAnalyzer(df)
                
                if not analyzer.analysis_results:
                    st.warning("⚠️ No analyzable feedback found. Please ensure your CSV has proper column headers and data.")
                    return
                
                # Store analyzer in session state
                st.session_state['analyzer'] = analyzer
                
                # Success message
                st.success("✅ Analysis complete! Explore your insights below.")
                
                # Display analysis results
                st.markdown("### 📊 Analysis Results")
                
                # Create two columns with better ratio
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Display results in expandable sections
                    for question, analysis in analyzer.analysis_results.items():
                        with st.expander(f"📝 {question}"):
                            st.markdown(f"**Total Responses:** {analysis['response_count']}")
                            
                            summary = analysis['summary']
                            if 'text_summary' in summary:
                                st.markdown("#### Key Insights")
                                st.write(summary['text_summary'])
                                
                                st.markdown("#### Common Themes")
                                if summary['common_themes']:
                                    themes_html = " ".join([
                                        f'<span class="theme-tag">{theme["word"]} ({theme["count"]})</span>'
                                        for theme in summary['common_themes']
                                    ])
                                    st.markdown(themes_html, unsafe_allow_html=True)
                                else:
                                    st.info("No common themes identified in the responses.")
                            
                            elif 'mean' in summary:
                                try:
                                    st.markdown("#### Statistical Overview")
                                    cols = st.columns(4)
                                    cols[0].metric("📊 Mean", f"{summary['mean']:.2f}")
                                    cols[1].metric("📈 Median", f"{summary['median']:.2f}")
                                    cols[2].metric("⬇️ Min", summary['min'])
                                    cols[3].metric("⬆️ Max", summary['max'])
                                except Exception:
                                    st.error("Could not display statistical information.")
                            
                            else:
                                try:
                                    st.markdown("#### Response Distribution")
                                    st.markdown(f"**Most Common:** {summary['most_common']}")
                                    for key, value in summary['distribution'].items():
                                        st.markdown(f"• {key}: {value} responses")
                                except Exception:
                                    st.error("Could not display distribution information.")
                
                with col2:
                    st.markdown("### 💬 Chat with Your Data")
                    st.markdown("Ask questions about your feedback and get instant insights!")
                    
                    # Example questions with better formatting
                    st.markdown("""
                    #### Try asking:
                    • 🔍 What are the main themes in the feedback?
                    • 📊 How many responses did we get?
                    • 💡 What did people say about [specific topic]?
                    • 📝 Give me a summary of all feedback.
                    """)
                    
                    # Chat input with friendly prompt
                    question = st.text_input("What would you like to know? 🤔")
                    if st.button("Get Insights 🔍") and question:
                        with st.spinner("Analyzing your question..."):
                            response = analyzer.answer_question(question)
                            st.markdown("#### Here's what I found:")
                            st.info(response)

        except Exception as e:
            st.error(f"🚨 Oops! Something went wrong: {str(e)}")
            st.markdown("Please check if your CSV file is properly formatted with headers and data.")
    else:
        # Show friendly welcome message when no file is uploaded
        st.info("👋 Welcome! Upload your feedback CSV file to get started. Need help? Check out the example questions on the right!")

if __name__ == "__main__":
    main() 
