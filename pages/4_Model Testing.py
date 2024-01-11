import streamlit as st 
import joblib 
import maps 
import data_validity
import model_dev
from fuzzywuzzy import process
import pandas as pd 
st.set_page_config(
    page_title="Model Testing",
    page_icon="🏰",
)

st.write("# 🏰 Welcome to Model Testing!")
st.header("Choose your trained model 🍯")
load_model = st.file_uploader("Load your model (Joblib file)", type="joblib")

if load_model is not None:
    loaded_model = joblib.load(load_model)
    st.success("Model loaded successfully")

    # Assuming `loaded_model` is the model object
    st.header("Model Information")
    model_name = data_validity.fuzzy_matching(list(maps.model_map.keys()), type(loaded_model.named_steps['model']).__name__)[0]
    st.subheader("Model Type")
    st.write(model_name)

    # Displaying model parameters
    st.subheader("Model Parameters")
    model_params = loaded_model.get_params()
    for param in model_params:
        if param in maps.model_map[model_name][2]:
            st.write(f"{param}: {model_params[param]}")
    
    st.subheader("Model Feature Importances")
    model_dev.feature_importances(model_name, loaded_model)


    st.header("Model Testing")
    load_test_data = st.file_uploader("Load cleaned test data (CSV) 🍯", type="csv")
    if load_test_data:
        test_data = pd.read_csv(load_test_data)
        st.dataframe(test_data.head())
        # Get y variable using fuzzy match 
        target_variable_input = st.text_input("Enter the Target Variable")
        target_variable, _ = process.extractOne(target_variable_input, test_data.columns)
        st.info(f"Selected Target Variable: {target_variable}")
        confirm_target = st.checkbox("I confirm the selected target variable")
        
        if confirm_target:
            # Initialize variables 
            y_test = test_data[target_variable]
            X_test = test_data.drop(columns=target_variable)

            exclude_input = st.text_input("Exclude Columns (type names, comma-separated)", "")
            if exclude_input:
                original_df = X_test.copy()

                # Deletes all columns entered from dataset 
                X_test = X_test.drop(columns = data_validity.fuzzy_matching(X_test, exclude_input))

                # Display an "Undo" button
                if st.button("Undo"):
                    X_test = original_df  # Revert to the original DataFrame

            # Visualizing training dataset 
            st.write("Preview of Transformed DataFrame")
            st.write(X_test.head())
            predictions = model_dev.predict_model(loaded_model, X_test)
            X_test["Prediction"] = predictions 
    
            # Display predictions 
            st.subheader("Model Predictions:")
            st.write(X_test)
            download_button = st.download_button(
            label="Download DataFrame with Predictions",
            key="download_preds",
            data=X_test.to_csv(index=False),
            file_name="preds_data.csv",
            mime="text/csv")
            