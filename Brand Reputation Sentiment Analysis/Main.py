import pandas as pd
import streamlit as st
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load the pretrained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Create the sentiment analysis pipeline
sentiment_analysis = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Define a function to calculate sentiment scores
def get_sentiment_scores(comments):
    scores = []
    for comment in comments:
        sentiment = sentiment_analysis(comment)[0]
        if sentiment['label'] == 'NEGATIVE':
            score = -1 * sentiment['score']
        else:
            score = sentiment['score']
        scores.append(score)
    return scores

# Set up Streamlit
st.title('Sentiment Analysis with DistilBERT')
uploaded_file = st.file_uploader("Upload an Excel file containing comments", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file, engine='openpyxl')

    if 'comment' not in df.columns:
        st.error('The Excel file should have a column named "comment".')
    else:
        comments = df['comment'].tolist()
        scores = get_sentiment_scores(comments)

        # Classify the comments and add scores to the DataFrame
        df['sentiment_score'] = scores
        df['sentiment'] = ['positive' if score > 0 else 'negative' for score in scores]

        # Display the DataFrame with sentiment scores and classification
        st.dataframe(df)

        # Calculate and display the average sentiment score
        avg_score = sum(scores) / len(scores)
        st.write(f'Average sentiment score: {avg_score:.2f}')
