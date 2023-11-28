import streamlit as st
import joblib  # Use joblib or your preferred library to load your pre-trained model

# Load your pre-trained XGBoost model here
model = joblib.load('cyber_bullying_XGB.pkl')  # Replace 'your_pretrained_model.pkl' with your model file path

# Define the Streamlit app
st.title("Cyberbullying Detection App")

# Add a text input widget for user input
user_input = st.text_input("Enter text:")

# Create a button to trigger the prediction
if st.button("Predict"):
    preprocessed_input = preprocess_input(user_input)  # Replace with your actual preprocessing code
    
    # Create a DMatrix object from the preprocessed input data
    dmatrix = xgb.DMatrix(preprocessed_input)
    
    # Perform the prediction
    prediction = model.predict(dmatrix)[0]  # Get the prediction result (assuming binary classification)

    # Display the prediction result to the user
    if prediction == 3:
        st.write("Cyberbullying Detected")
    else:
        st.write("No Cyberbullying Detected")
