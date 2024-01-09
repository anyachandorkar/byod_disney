import streamlit as st
import pandas as pd
import Home
import data_validity
import model_dev
from sklearn.model_selection import train_test_split
from fuzzywuzzy import process
import numpy as np

models = ["", "Random Forest (Classifier)",
        "XGBoost (Classifier)",
        "Logistic Regression",
        "K-Nearest Neighbors (Classifier)"]

eval_metrics = [
    'Accuracy',
    'Precision',
    'Recall',
    'Confusion Matrix',
    'F1 Score']

# Setting session state 
session_state = st.session_state
if not hasattr(session_state, 'cls_model_history'):
    session_state.cls_model_history = {}

st.header("Choose your cleaned CSV file 🍯")
cls_file = st.file_uploader("Choose a CSV file", type="csv")

if cls_file: 
    # Clears model history if new dataset is uploaded 
    if not hasattr(session_state, 'cls_current_dataset') or session_state.cls_current_dataset != cls_file:
        session_state.cls_model_history = {}
        session_state.cls_current_dataset = cls_file

    # Visualizing uploaded dataset 
    df = pd.read_csv(cls_file)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns = ['Unnamed: 0'])
    st.dataframe(df.head())
    
    # Choosing variable to predict 
    target_variable_input = st.text_input("Enter the Target Variable for Classification:")
    target_variable, _ = process.extractOne(target_variable_input, df.columns)
    st.info(f"Selected Target Variable: {target_variable}")
    confirm_target = st.checkbox("I confirm the selected target variable")

    if confirm_target:
        # Checking dataset for any imbalance to warn user before training
        st.header('Data Imbalance Check') 
        imbalance_check = data_validity.check_imbalance(df[target_variable])
        st.warning(f"Imbalance Check: {imbalance_check}")
        if imbalance_check:
            st.warning("You may want to use downsampling or SMOTE techniques to balance your dataset before training.")
       
        # Plotting distribution of target variable 
        st.subheader("Visualize Target Variable Distribution")
        data_validity.visualize_target_variable(df, target_variable, "Classification")
        
        st.header("Feature Selection")
        multicollinear_vars = data_validity.check_multicollinearity(df)
        if multicollinear_vars[0]:
            st.warning('Multicollinearity Check: True')
            st.warning('Consider streamlining dataset based on highly correlated feature pairs')
            st.subheader('Correlated Features:')
            for pair in multicollinear_vars[0]:
                st.write(pair)
            st.pyplot(multicollinear_vars[1])
        else:
            st.warning('Multicollinearity Check: False')
            st.write('No highly correlated features to streamline')

        # Providing option to exclude columns from training dataset 
        st.subheader("Data Subsetting")
        exclude_input = st.text_input("Exclude Columns (type names, comma-separated)", "")
        if exclude_input:
            original_df = df.copy()

            # Deletes all columns entered from dataset 
            df = df.drop(columns = data_validity.fuzzy_matching(df, exclude_input))

            # Display an "Undo" button
            if st.button("Undo"):
                df = original_df  # Revert to the original DataFrame

        # Visualizing training dataset 
        st.write("Preview of Transformed DataFrame")
        st.write(df.head())
        
        text_cols_input = st.checkbox("Do you have any text columns?")
        if text_cols_input:
            user_input = st.text_area("List the text columns (comma-separated)", "")
            matched_columns = data_validity.fuzzy_matching(df, user_input)
            st.write("Matched text columns:", matched_columns)
            st.write(type(matched_columns))
            st.write(df[matched_columns])
        else:
            matched_columns = []

        # Option to select model 
        st.header("Model Testing")

        # Providing adjustable train test split options 
        test_size = st.slider("Select Test Size:", min_value=0.1, max_value=0.5, value=0.2, step=0.01)
        confirm_param = st.checkbox("Apply hyperparameter tuning")
        user_model = st.selectbox("Choose Your Model", list(models))

        if user_model != "": 
            # Once model is chosen separate features and target variable 
            X = df.drop(columns = [target_variable])
            y = df[target_variable]

            # Apply train test split to data using previously stored test_size variable 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Preprocessing input of data to be read by model 
            preprocessor = model_dev.preprocess_data(X_train, matched_columns)
            
            # Fitting model 
            trained_model = model_dev.fit_model(X_train, y_train, preprocessor, user_model, confirm_param)
           
            # Getting model predictions 
            predictions = model_dev.predict_model(trained_model, X_test)
            
            # Allowing users to choose their evaluation metric 
            metric = st.selectbox("Choose your evaluation metric:", eval_metrics)
            num_classes = len(np.unique(y_train))
            if num_classes>2:
                results = model_dev.evaluate_model(y_test, predictions, metric, X, average='weighted')
            else:
                results = model_dev.evaluate_model(y_test, predictions, metric, X)
            st.write(results)

            if metric!='Confusion Matrix':
                if results>0.70:
                    st.success('Nice performance!')
                else: 
                    st.warning('You may want to fine tune your model or try another one.')
            
            # Displaying model history with performance for comparison 
            st.subheader("Model History")
            session_state.cls_model_history[user_model] = {'metric': metric, 'result': results}
            for model, history in session_state.cls_model_history.items():
                st.write(f"{model}: Metric={history['metric']}, Result={history['result']}")
            
            # Running and visualizing feature importances if applicable to model 
            st.header("Feature Importance")
            model_dev.feature_importances(user_model, trained_model)
        