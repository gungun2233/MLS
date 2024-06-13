import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import joblib

# Load dataset
file_path = 'placement.csv'
dataset = pd.read_csv(file_path)

# Features and target
X = dataset[['cgpa', 'resume_score']]
y = dataset['placed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Perceptron model
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(perceptron, 'perceptron_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the model and scaler
perceptron_model = joblib.load('perceptron_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title('Placement Prediction using Perceptron Model')

st.write('Enter the CGPA and Resume Score to predict placement:')

# User input
cgpa = st.number_input('CGPA', min_value=0.0, max_value=10.0, value=0.0)
resume_score = st.number_input('Resume Score', min_value=0.0, max_value=10.0, value=0.0)

# Prepare the input data for prediction
input_data = pd.DataFrame([[cgpa, resume_score]], columns=['cgpa', 'resume_score'])
input_data = scaler.transform(input_data)

# Make prediction
prediction = perceptron_model.predict(input_data)

# Display the prediction result
if st.button('Predict'):
    if prediction[0] == 1:
        st.success('The candidate is likely to be placed.')
    else:
        st.error('The candidate is not likely to be placed.')
