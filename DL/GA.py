import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

# Load the dataset
st.title("Major League Soccer (MLS) Prediction App")

# Assuming the dataset is in the same directory as the app.py file
df = pd.read_csv("GA.csv")

# Data preprocessing
df.drop('Serial No.', axis=1, inplace=True)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

# Get the input feature ranges
feature_ranges = {
    feature: (float(X[feature].min()), float(X[feature].max()))
    for feature in X.columns
}

# Get input from the user
input_features = []
for feature in X.columns:
    min_value, max_value = feature_ranges[feature]
    feature_value = st.number_input(
        f"Enter value for {feature} (range: {min_value} - {max_value})",
        min_value=float(min_value),
        max_value=float(max_value),
        value=float((min_value + max_value) / 2),
    )
    input_features.append(feature_value)

# Make predictions when the button is clicked
if st.button("Predict"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    SCALAR = MinMaxScaler()
    X_train_scaled = SCALAR.fit_transform(X_train)
    X_test_scaled = SCALAR.transform(X_test)

    # Build the model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='Adam')

    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

    input_data = np.array([input_features])
    input_data_scaled = SCALAR.transform(input_data)
    prediction = model.predict(input_data_scaled)

    st.write("Prediction:")
    st.write(prediction[0][0])

    # Interpret the output and display balloon
    if prediction[0][0] > 0.7:
        st.write("Based on the input features, the predicted outcome is positive.")
        st.balloons()
    elif prediction[0][0] < 0.3:
        st.write("Based on the input features, the predicted outcome is negative.")
        st.snow()
    else:
        st.write("Based on the input features, the predicted outcome is neutral.")