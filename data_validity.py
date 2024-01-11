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
    '''
    If the minority class in target variable constitutes less than 25% of the total 
    function flags as imbalance
    '''
    class_counts = series.value_counts()
    minority_class = class_counts.idxmin()
    minority_percentage = class_counts.min() / class_counts.sum()
    return minority_percentage < threshold

# Function to visualize target variable distribution based on problem type
def visualize_target_variable(df, target_variable, problem_type):

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
    if type(df)!=list:
        column_names = df.columns.tolist()
    else:
        column_names = df
    # Use fuzzy matching to find the closest matches to the user input
    matches = process.extractBests(user_input, column_names)

    # Select column names with a match score above a certain threshold (e.g., 80)
    threshold = 80
    selected_columns = [match[0] for match in matches if match[1] > threshold]

    return selected_columns

def check_multicollinearity(df, threshold=0.69):
    """
    Check multicollinearity among predictor variables in a DataFrame. If there are feature pairs 
    that have correlation coefficient greater than 0.69 threshold, they will be returned
    """
    # Exclude non-numeric columns
    numeric_vars = df.select_dtypes(include=['number']).columns

    # Calculate correlation matrix
    correlation_matrix = df[numeric_vars].corr()

    # Extract pairs of highly correlated variables
    multicollinear_vars = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                multicollinear_vars.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
    plt.figure(figsize=(4, 3))
    sns.heatmap(df[numeric_vars].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    return [multicollinear_vars, plt]