import streamlit as st
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained XGBoost model
model = joblib.load('cyber_bullying_XGB.pkl')

# Define a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Define the preprocess_input function
def preprocess_input(user_input):
    # Fit and transform the TF-IDF vectorizer on your training data
    tfidf_vectorizer.fit(training_data)  # Replace 'training_data' with your actual training data
    tfidf_matrix = tfidf_vectorizer.transform([user_input])
    return tfidf_matrix

# Define the Streamlit app
st.title("Cyberbullying Detection App")

# Add a text input widget for user input
user_input = st.text_input("Enter text:")

# Create a button to trigger the prediction
if st.button("Predict"):
    # Preprocess the user input
    preprocessed_input = preprocess_input(user_input)
    
    # Create a DMatrix object from the preprocessed input data
    dmatrix = xgb.DMatrix(preprocessed_input)
    
    # Perform the prediction
    prediction = model.predict(dmatrix)[0]  # Get the prediction result (assuming binary classification)

    # Display the prediction result to the user
    if prediction == 3:
        st.write("Cyberbullying Detected")
    else:
        st.write("No Cyberbullying Detected")
