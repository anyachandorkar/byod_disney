import streamlit as st
import pandas as pd
import Home
import data_validity
import model_dev
from fuzzywuzzy import process

st.set_option('deprecation.showPyplotGlobalUse', False)
models = ["K-Means",
    "DBSCAN",
    "GMM",
    "Agglomerative"]

st.header("Choose your cleaned CSV file")
reg_file = st.file_uploader("Choose a CSV file", type="csv")
if reg_file: 
    # Visualizing uploaded dataset 
    df = pd.read_csv(reg_file)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns = ['Unnamed: 0'])
    st.dataframe(df.head())

    # Providing option to exclude columns from training dataset 
    exclude_input = st.text_input("Exclude Columns (type names, comma-separated)", "")
    if exclude_input:
        original_df = df.copy()

        # Deletes all columns entered from dataset 
        df = df.drop(columns = data_validity.fuzzy_matching(df, exclude_input))

        # Display an "Undo" button
        if st.button("Undo"):
            df = original_df  # Revert to the original DataFrame
            
    st.write("Preview of Transformed DataFrame")
    st.write(df.head())

    # Option to select model 
    st.subheader("Model Selection")
    user_model = st.selectbox("Choose Your Model", list(models))
    if user_model != "":
        k=None
        if user_model in ["K-Means","GMM","Agglomerative"]:
            k = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=3)
        preprocessor = model_dev.preprocess_data(df)
        # Run clustering
        if st.button("Run Clustering"):
            trained_model = model_dev.fit_clusters(preprocessor.fit_transform(df), user_model, k)
            
            # Show predictions 
            st.write("Clustering Predictions:")
            df['Cluster'] = trained_model
            st.write(df)
            download_button = st.download_button(
            label="Download DataFrame with Clusters",
            key="download_clusters",
            data=df.to_csv(index=False),
            file_name="clusters_data.csv",
            mime="text/csv")

            # Show clusters
            model_dev.visualize_clusters(preprocessor.fit_transform(df), trained_model)