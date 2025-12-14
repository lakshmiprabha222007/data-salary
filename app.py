import streamlit as st
import pickle
import numpy as np

# Set the correct model file name from your snippet
MODEL_FILE = "trained_salary_LR_model.pkl"

# --- Model Loading ---
model = None
try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    
    # Optional: Check if the model is a scikit-learn model and has the expected feature count
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != 1:
        st.warning(f"Model loaded expects {model.n_features_in_} features, but the app is configured for 1 feature.")
    
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_FILE}. Please ensure it is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


st.title("Salary Prediction App")
st.subheader("Predict Salary based on Years of Experience")

# --- Single Input Field ---
# The single feature is Year of Experience
input_feature = st.number_input(
    "Enter Years of Experience:", 
    min_value=0.0, 
    value=5.0, 
    step=0.1
)

if st.button("Predict Salary"):
    # Reshape the single input value into the 2D array format required by scikit-learn: [[value]]
    input_data = np.array([[input_feature]])
    
    try:
        prediction = model.predict(input_data)[0]
        
        # Format the prediction as currency for better display
        formatted_salary = f"${prediction:,.2f}"
        
        st.success(f"Predicted Salary: {formatted_salary}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
