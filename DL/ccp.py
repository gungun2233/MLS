import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Load the dataset
df = pd.read_csv(r"C:\Users\Asus\OneDrive\Desktop\MLS\MLS\DL\ccp.csv")

# Preprocess the data
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)  # One-hot encode categorical features
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Split the data into features and target
X = df.drop(columns=['Exited'])
y = df['Exited'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(11, activation='sigmoid', input_dim=11))
model.add(Dense(11, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_trf, y_train, batch_size=50, epochs=10, verbose=1, validation_split=0.2)

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
st.title("Customer Churn Prediction")

# Sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    "1. Enter the customer details in the provided input fields.\n"
    "2. Click the 'Predict' button to get the churn prediction."
)

# Function to get user input
def user_input_features():
    st.subheader("Enter Customer Details")
    CreditScore = st.number_input('Credit Score', min_value=300, max_value=850, value=600, step=1, key='credit_score', help="Suggestion: Higher credit scores generally indicate lower churn risk.")
    Age = st.number_input('Age', min_value=18, max_value=100, value=30, step=1, key='age', help="Suggestion: Different age groups may have varying churn tendencies.")
    Tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5, step=1, key='tenure', help="Suggestion: Longer tenure often correlates with lower churn risk.")
    Balance = st.number_input('Balance', min_value=0.0, max_value=1e8, value=50000.0, step=100.0, key='balance', help="Suggestion: Customers with higher account balances may be less likely to churn.")
    NumOfProducts = st.number_input('Number of Products', min_value=1, max_value=4, value=2, step=1, key='num_products', help="Suggestion: Customers with more products may be more engaged and less likely to churn.")
    HasCrCard = st.selectbox('Has Credit Card', [0, 1], key='has_cr_card', help="Suggestion: Credit card holders may have different churn patterns.")
    IsActiveMember = st.selectbox('Is Active Member', [0, 1], key='is_active_member', help="Suggestion: Active members may be less likely to churn.")
    EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0, max_value=1e8, value=50000.0, step=1000.0, key='estimated_salary', help="Suggestion: Higher salaries may indicate greater financial stability and lower churn risk.")
    Geography_Germany = st.selectbox('Geography Germany', [0, 1], key='geography_germany', help="Suggestion: Geographical factors can influence customer behavior and churn.")
    Geography_Spain = st.selectbox('Geography Spain', [0, 1], key='geography_spain', help="Suggestion: Geographical factors can influence customer behavior and churn.")
    Gender_Male = st.selectbox('Gender Male', [0, 1], key='gender_male', help="Suggestion: Gender may impact customer preferences and churn tendencies.")

    data = {
        'CreditScore': CreditScore,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary,
        'Geography_Germany': Geography_Germany,
        'Geography_Spain': Geography_Spain,
        'Gender_Male': Gender_Male
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
user_input = user_input_features()

# Predict button
if st.button("Predict"):
    # Standardize user input features
    input_trf = scaler.transform(user_input)

    # Predict churn
    prediction = model.predict(input_trf)

    # Display prediction
    st.subheader('Prediction')
    if prediction[0, 0] > 0.5:
        st.error("Customer is likely to churn. 😔")
    else:
        st.success("Customer is likely to stay. 😃")
        st.balloons()

# About section
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This app predicts customer churn using a deep learning model trained on historical customer data. "
    "It considers various features like credit score, age, tenure, balance, and more to estimate the probability of a customer leaving."
)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by [Gungun sharma]")
