import streamlit as st
import pandas as pd
import Home
import data_validity
import model_dev
from sklearn.model_selection import train_test_split
from fuzzywuzzy import process

models = ["Random Forest (Classifier)",
        "XGBoost (Classifier)",
        "SVM (Classifier)",
        "Logistic Regression",
        "K-Nearest Neighbors (Classifier)"]
eval_metrics = ['Confusion Matrix',
    'Accuracy',
    'Precision',
    'Recall',
    'ROC Curve',
    'Precision-Recall Curve',
    'AUC (ROC)',
    'AUC (Precision-Recall)',
    'F1 Score']

st.header("Choose your cleaned CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
df = pd.read_csv(uploaded_file).drop(columns = ['Unnamed: 0'])
st.dataframe(df.head())
target_variable_input = st.text_input("Enter the Target Variable for Classification:")
target_variable, _ = process.extractOne(target_variable_input, df.columns)
st.info(f"Selected Target Variable: {target_variable}")
confirm_target = st.checkbox("I confirm the selected target variable")

if confirm_target:
    imbalance_check = data_validity.check_imbalance(df[target_variable])
    st.warning(f"Imbalance Check: {imbalance_check}")
    if imbalance_check:
        st.warning("You may want to use downsampling or SMOTE techniques to balance your dataset before training.")
    data_validity.visualize_target_variable(df, target_variable, "Classification")
    st.header("Train-Test Split Options")
    test_size = st.slider("Select Test Size:", min_value=0.1, max_value=0.5, value=0.2, step=0.01)
    
    exclude_input = st.text_input("Exclude Columns (type names, comma-separated)", "")
    if exclude_input:
        df = data_validity.exclude_columns(df, exclude_input)
    st.write("Preview of Transformed DataFrame")
    st.write(df.head())

    user_model = st.selectbox("Choose Your Model", list(models))
    if user_model: 
        X = df.drop(columns = [target_variable])
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        num_columns, cat_columns, preprocessor = model_dev.preprocess_data(X_train)

        trained_model = model_dev.fit_model(X_train, y_train, preprocessor, user_model)
        predictions = model_dev.predict_model(trained_model, X_test)
        st.write(predictions)
        metric = st.selectbox("Choose your evaluation metric:", eval_metrics)
        results = model_dev.evaluate_model(y_test, predictions, metric)
        st.write(results)
    