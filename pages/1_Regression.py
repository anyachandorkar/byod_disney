import streamlit as st
import pandas as pd
import data_validity
import model_dev

models = ["Random Forest (Regressor)",
        "XGBoost (Regressor)",
        "SVM (Regressor)",
        "Linear Regression",
        "K-Nearest Neighbors (Regressor)"]

df = return_data()
target_variable = st.selectbox("Choose the Target Variable for Regression:", df.select_dtypes(include=['number']).columns)
outlier_check = data_validity.check_outliers(df[target_variable])
st.warning(f"Outlier Check: {outlier_check}")