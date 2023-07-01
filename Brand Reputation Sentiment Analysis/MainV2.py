import re
import pandas as pd
import streamlit as st
import numpy as np
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration



# Load the pretrained DistilBERT model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Create the sentiment analysis pipeline
sentiment_analysis = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Load the pretrained T5 model and tokenizer for advice generation
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define a function to clean the comments
def clean_comment(comment):
    # Remove non-alphanumeric characters
    comment = str(comment)
    cleaned_comment = re.sub(r'\W+', ' ', comment)
    return cleaned_comment.strip()

# Define a function to generate advice based on negative comments
def generate_advice(negative_comments):
    input_text = f"summarize: {negative_comments}"
    input_tokens = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    output_tokens = t5_model.generate(input_tokens, max_length=512, num_return_sequences=1)
    advice = t5_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return advice

# Set up Streamlit
st.title('Sentiment Analysis and Advice Generation with DistilBERT and T5')
uploaded_file = st.file_uploader("Upload a CSV or Excel file containing comments", type=['csv', 'xlsx'])

if uploaded_file:

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')


    if 'comment' not in df.columns:
        st.error('The Excel file should have a column named "comment".')
    else:
        # Clean the comments
        df = df[df['comment'].notna()]
        df['cleaned_comment'] = df['comment'].apply(clean_comment)
        comments = df['cleaned_comment'].tolist()
        sentiment_results = sentiment_analysis(comments)

        # Classify the comments and add scores to the DataFrame
        df['sentiment_score'] = [result['score'] if result['label'] == 'POSITIVE' else -result['score'] for result in sentiment_results]
        df['sentiment'] = [result['label'].lower() for result in sentiment_results]

        # Display the DataFrame with sentiment scores and classification
        st.dataframe(df)

        # Calculate and display the average sentiment score
        pos = 0
        neg = 0
        for i in df['sentiment']:
            if i == 'positive':
                pos = pos + 1
            elif i == 'negative':
                neg = neg + 1
        pos = pos
        neg = neg
        tot = len(df['sentiment'])
        st.write(f'Number of records are: {tot}')
        st.write(f'Number of Positive comments are: {pos}')
        st.write(f'Number of Negative comments are: {neg}')
        avg_score = df['sentiment_score'].mean()
        st.write(f'Average sentiment score: {avg_score:.2f}')

        # Generate advice based on overall negative comments
        negative_comments = ' '.join(df[df['sentiment'] == 'negative']['cleaned_comment'].tolist())
        advice = generate_advice(negative_comments)
        st.write(f'Suggested improvements based on negative comments: {advice}')

        st.area_chart(data=df,x='sentiment',y='sentiment_score')
        st.line_chart(data=df,x='sentiment',y='sentiment_score')




