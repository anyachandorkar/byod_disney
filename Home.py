import streamlit as st
import pandas as pd 

st.set_page_config(
    page_title="BYOD",
    page_icon="🏰",
)

st.write("# Welcome to Bring Your Own Data! 👋")
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
            st.warning("Warning: The dataset contains missing values. Please clean the dataset.")
        else:
            st.success("The dataset appears to be clean and ready for analysis. Please choose a problem type in the sidebar.")

    except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Please ensure that the uploaded file is a valid CSV.")
