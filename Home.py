import streamlit as st
import pandas as pd 
import streamlit as st
from fuzzywuzzy import process
import data_validity

'''
Code for Home page 
Includes 
    - Null value check 
    - Stats and feature based visualization 
'''

st.set_page_config(page_title="BYOD", page_icon="🏰")

st.write("# Welcome to Bring Your Own Data! 🍯")
st.header("Upload your CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Display the first few rows of the dataset
        st.dataframe(df.head())

        # Check for missing values
        if df.isnull().any().any():
            st.warning("Warning: The dataset contains missing values. Please clean the dataset before proceeding.")
        else:
            st.success("The dataset appears to be clean and ready for analysis. You may choose a problem type in the sidebar or explore your data below.")
            
            # Display data stats 
            st.subheader('Dataset Statistics:')
            st.dataframe(df.describe())

            # Allow user input based feature visualization using fuzzy matching 
            st.subheader('Visualize Feature Distributions')
            feature_input = st.text_input("Enter the feature for visualization:")
            feature_variable, _ = process.extractOne(feature_input, df.columns)
            st.info(f"Selected Target Variable: {feature_variable}")
            confirm_target = st.checkbox("I confirm the selected feature variable")
            if confirm_target: 
                if feature_variable in df.select_dtypes(include=['int', 'float']).columns.tolist():
                    data_validity.visualize_target_variable(df, feature_variable, 'Regression')
                else:
                    data_validity.visualize_target_variable(df, feature_variable, 'Classification')
    except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Please ensure that the uploaded file is a valid CSV.")
