import streamlit as st 
import joblib 

load_model = st.file_uploader("Load Model (Joblib file)", type="joblib")
if load_model is not None:
    loaded_model = joblib.load(load_model)
    st.success("Model loaded successfully")
    # Assuming `loaded_model` is the model object
    st.subheader("Model Information")
    st.write(f"Model Type: {type(loaded_model).__name__}")
    # Display other relevant information about the model
