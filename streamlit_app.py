import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

download_nltk_data()

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
        
        # Tokenize sentences
        sentences = sent_tokenize(combined_text)
        
        if len(sentences) == 0:
            return {
                'text_summary': "No text responses available.",
                'common_themes': [],
                'total_responses': 0
            }
        
        # Calculate sentence importance using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Get average TF-IDF scores for each sentence
        importance_scores = tfidf_matrix.sum(axis=1).A1
        
        # Get top sentences (most representative)
        top_sentence_indices = importance_scores.argsort()[-3:][::-1]
        summary_sentences = [sentences[i] for i in top_sentence_indices]
        
        # Create a summary paragraph
        summary = " ".join(summary_sentences)
        
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
        # Tokenize and clean text
        stop_words = set(stopwords.words('english'))
        words = ' '.join(responses).lower().split()
        words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequencies
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Get top 5 most common words
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{'word': word, 'count': count} for word, count in common_words]

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
        # Define common question patterns and their handling logic
        if "how many" in question.lower() and "responses" in question.lower():
            total_responses = max(data['response_count'] for data in self.analysis_results.values())
            return f"There are {total_responses} total responses in the feedback data."
        
        if "common themes" in question.lower() or "main themes" in question.lower():
            themes = []
            for data in self.analysis_results.values():
                if 'text_summary' in data['summary']:
                    themes.extend(theme['word'] for theme in data['summary']['common_themes'])
            if themes:
                return f"The main themes identified across all responses are: {', '.join(set(themes))}."
            
        if "summary" in question.lower():
            return self.feedback_context
        
        # For questions about specific questions/topics, search through the data
        for question_text, data in self.analysis_results.items():
            if any(word in question_text.lower() for word in question.lower().split()):
                if 'text_summary' in data['summary']:
                    return f"Regarding '{question_text}': {data['summary']['text_summary']}"
                elif 'mean' in data['summary']:
                    return f"For '{question_text}', the average is {data['summary']['mean']:.2f} (range: {data['summary']['min']} to {data['summary']['max']})"
                else:
                    return f"For '{question_text}', the most common response was '{data['summary']['most_common']}'"
        
        return "I'm not sure about that specific aspect of the feedback. Could you rephrase your question or ask about a specific question from the survey?"

def main():
    st.set_page_config(
        page_title="Event Feedback Analyzer",
        page_icon="📊",
        layout="wide"
    )

    st.title("Event Feedback Analyzer 📊")
    st.write("Upload your Google Forms feedback CSV file and get instant analysis and insights!")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Create analyzer instance
            analyzer = FeedbackAnalyzer(df)
            
            # Store analyzer in session state
            st.session_state['analyzer'] = analyzer
            
            # Display analysis results
            st.subheader("📈 Analysis Results")
            
            # Create two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display results in expandable sections
                for question, analysis in analyzer.analysis_results.items():
                    with st.expander(f"📝 {question}"):
                        st.write(f"Total Responses: {analysis['response_count']}")
                        
                        summary = analysis['summary']
                        if 'text_summary' in summary:
                            st.write("Summary:")
                            st.write(summary['text_summary'])
                            
                            st.write("Common Themes:")
                            themes_html = " ".join([
                                f'<span style="background-color: #e9ecef; padding: 4px 8px; margin: 2px; border-radius: 4px;">{theme["word"]} ({theme["count"]})</span>'
                                for theme in summary['common_themes']
                            ])
                            st.markdown(themes_html, unsafe_allow_html=True)
                        
                        elif 'mean' in summary:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean", f"{summary['mean']:.2f}")
                            col2.metric("Median", f"{summary['median']:.2f}")
                            col3.metric("Min", summary['min'])
                            col4.metric("Max", summary['max'])
                        
                        else:
                            st.write(f"Most Common Response: {summary['most_common']}")
                            st.write("Distribution:")
                            for key, value in summary['distribution'].items():
                                st.write(f"- {key}: {value} responses")
            
            with col2:
                st.subheader("💬 Ask Questions")
                st.write("Ask me anything about the feedback!")
                
                # Example questions
                st.markdown("""
                Try asking:
                - What are the main themes in the feedback?
                - How many responses did we get?
                - What did people say about [specific topic]?
                - Give me a summary of all feedback.
                """)
                
                # Chat input
                question = st.text_input("Your question:")
                if st.button("Ask") and question:
                    response = analyzer.answer_question(question)
                    st.write("Answer:")
                    st.info(response)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 