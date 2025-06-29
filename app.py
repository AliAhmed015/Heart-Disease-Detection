import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained models
with open('lr_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

# Load the scaler if available
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App title and description
st.write("""
# Heart Disease Prediction App

This app predicts the **presence of heart disease** based on user input features.

Data obtained from a heart disease dataset.
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 50)
    resting_bp = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
    cholesterol = st.sidebar.slider('Cholesterol', 100, 400, 200)
    max_hr = st.sidebar.slider('Max Heart Rate Achieved', 60, 200, 150)
    st_depression = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0)
    num_major_vessels = st.sidebar.slider('Number of Major Vessels', 0, 3, 1)
    resting_ecg = st.sidebar.selectbox('Resting ECG', (0, 1, 2))
    slope = st.sidebar.selectbox('Slope', (0, 1, 2))
    fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar', (0, 1))
    exercise_angina = st.sidebar.selectbox('Exercise Induced Angina', (0, 1))
    thalassemia = st.sidebar.selectbox('Thalassemia', (0, 1, 2, 3))
    chest_pain = st.sidebar.selectbox('Chest Pain Type', (0, 1, 2, 3))
    sex = st.sidebar.selectbox('Sex', (0, 1))

    data = {
        'Age': age,
        'RestingBloodPressure': resting_bp,
        'Cholesterol': cholesterol,
        'MaxHeartRateAchieved': max_hr,
        'STDepression': st_depression,
        'NumberOfMajorVessels': num_major_vessels,
        'RestingECG': resting_ecg,
        'Slope': slope,
        'FastingBloodSugar': fasting_bs,
        'ExerciseInducedAngina': exercise_angina,
        'Thalassemia': thalassemia,
        'ChestPainType': chest_pain,
        'Sex': sex
    }

    features = pd.DataFrame(data, index=[0])
    return features

# User input features
input_df = user_input_features()

# Display the user input features
st.subheader('User Input Features')
st.write(input_df)

# Scale the input features
inputs_array = input_df.values
scaled_inputs_array = scaler.transform(inputs_array)

# Apply models to make predictions
inputs_array = input_df.values

# Logistic Regression Prediction
lr_prediction = lr_model.predict(scaled_inputs_array)[0]
lr_prediction_proba = lr_model.predict_proba(scaled_inputs_array)[0]

# Display predictions
st.subheader('Logistic Regression Prediction')
st.write(lr_prediction)
st.write(lr_prediction_proba[1])
