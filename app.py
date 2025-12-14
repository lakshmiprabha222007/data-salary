import streamlit as st
import pickle
import numpy as np

# Load trained Linear Regression model
with open("trained_salary_LR_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Salary Prediction App")

# Detect number of features automatically
1_features = model.1_features_in_

st.write(f"Model expects {1_features} input feature(s)")

# Input fields
inputs = []
for i in range(n_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

if st.button("Predict Salary"):
    input_data = np.array([inputs])
    prediction = model.predict(input_data)
    st.success(f"Predicted Salary: {prediction[0]}")
