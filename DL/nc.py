import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the model architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Make predictions on the test set
y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)

# Streamlit app
st.title("MNIST Digit Recognition")

# Get the index input from the user
sample_index = st.number_input("Enter an index (0-9999)", min_value=0, max_value=9999, value=0, step=1)

# Ensure the index is within the valid range
if sample_index >= len(X_test):
    st.write(f"Index {sample_index} is out of range. Please enter a valid index between 0 and {len(X_test) - 1}.")
else:
    sample_image = X_test[sample_index]

    if st.button("Predict"):
        st.image(sample_image.reshape(28, 28))
        predicted_digit = y_pred[sample_index]
        st.write(f"The predicted digit is: {predicted_digit}")