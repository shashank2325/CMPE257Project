import streamlit as st
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained XGBoost model
model = joblib.load('cyber_bullying_XGB.pkl')

# Load your dataset (replace 'dataset.csv' with the actual file path)
dataset = pd.read_csv('cyberbullying_tweets.csv')  # Replace with your dataset file path

# Extract the tweet_text column as training data
training_data = dataset['tweet_text'].tolist()

# Define a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Define the preprocess_input function
def preprocess_input(user_input):
    # Fit and transform the TF-IDF vectorizer on your training data
    tfidf_vectorizer.fit(training_data)
    user_input_vectorized = tfidf_vectorizer.transform([user_input])
    return user_input_vectorized

# Define the Streamlit app
st.title("Cyberbullying Detection App")

# Add a text input widget for user input
user_input = st.text_input("Enter text:")

# Create a button to trigger the prediction
if st.button("Predict"):
    # Preprocess the user input
    preprocessed_input = preprocess_input(user_input)
    
    # Perform the prediction
    prediction = model.predict(preprocessed_input)[0]  # Get the prediction result (assuming binary classification)

    # Display the prediction result to the user
    if prediction == 1:
        st.write("Cyberbullying Detected")
    else:
        st.write("No Cyberbullying Detected")
