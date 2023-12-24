import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fuzzywuzzy import process

def check_outliers(series, threshold=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return any((series < lower_bound) | (series > upper_bound))

# Function to check for imbalance in a categorical series
def check_imbalance(series, threshold=0.25):
    class_counts = series.value_counts()
    minority_class = class_counts.idxmin()
    minority_percentage = class_counts.min() / class_counts.sum()
    return minority_percentage < threshold

# Function to visualize target variable distribution based on problem type
def visualize_target_variable(df, target_variable, problem_type):
    st.header("Visualize Target Variable Distribution")

    if problem_type == "Classification":
        # Bar chart for classification
        plt.figure(figsize=(2, 1))
        sns.countplot(x=target_variable, data=df)
        plt.title("Class Distribution")
        plt.xlabel(target_variable)
        plt.ylabel("Count")
        st.pyplot(plt)

    else:
        # Histogram for regression
        plt.figure(figsize=(3, 2))
        sns.histplot(df[target_variable])
        plt.title("Target Variable Distribution")
        plt.xlabel(target_variable)
        plt.ylabel("Frequency")
        st.pyplot(plt)

def fuzzy_matching(df, user_input):
    # Get a list of column names from the DataFrame
    column_names = df.columns.tolist()

    # Use fuzzy matching to find the closest matches to the user input
    matches = process.extractBests(user_input, column_names)

    # Select column names with a match score above a certain threshold (e.g., 80)
    threshold = 80
    selected_columns = [match[0] for match in matches if match[1] > threshold]

    return selected_columns

def text_column_extraction(df, user_input):
    user_text_columns = [col.strip() for col in user_input.split(',')]
    return user_text_columns
    # Use fuzzy matching to find the closest matches in the dataset
    available_columns = df.columns.tolist()
    matched_columns, _ = process.extract(user_text_columns, available_columns)
    return matched_columns