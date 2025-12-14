import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("trained_salary_LR_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Salary Prediction App")

# Single input feature
feature = st.number_input("Enter Years of Experience", value=0.0)

if st.button("Predict Salary"):
    input_data = np.array([[feature]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Salary: {prediction[0]}")

