import streamlit as st
import pandas as pd
import Home
import data_validity
import model_dev
from fuzzywuzzy import process
from collections import defaultdict 

'''
Code for Clustering Tab
Includes: 
    - Multicollinearity check 
    - Data subsetting 
    - Model selection and evaluation 
    - Model history 
    - Data with cluster download 
    - Cluster visualization  
    - Model deployment 

'''

st.set_page_config(
    page_title="Clustering",
    page_icon="🏰",
)

st.set_option('deprecation.showPyplotGlobalUse', False)
models = ["","K-Means",
    "DBSCAN",
    "GMM",
    "Agglomerative"]

# Setting session state 
session_state = st.session_state
if not hasattr(session_state, 'clu_model_history'):
    session_state.clu_model_history = {}

st.header("Choose your cleaned CSV file 🍯")
clu_file = st.file_uploader("Choose a CSV file", type="csv")

if clu_file: 
    # Clears model history if new dataset is uploaded 
    if not hasattr(session_state, 'clu_current_dataset') or session_state.clu_current_dataset != clu_file:
        session_state.clu_model_history = {}
        session_state.clu_current_dataset = clu_file

    # Visualizing uploaded dataset 
    df = pd.read_csv(clu_file)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns = ['Unnamed: 0'])
    st.dataframe(df.head())

    st.header("Feature Selection")
    multicollinear_vars = data_validity.check_multicollinearity(df)
    if multicollinear_vars[0]:
        st.warning('Multicollinearity Check: True')
        st.subheader('Correlated Features:')
        for pair in multicollinear_vars[0]:
            st.write(pair)
        st.pyplot(multicollinear_vars[1])
        st.warning('Consider streamlining dataset based on highly correlated feature pairs')
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
            
    st.write("Preview of Transformed DataFrame")
    st.write(df.head())

    # Option to select model 
    st.header("Model Testing")
    user_model = st.selectbox("Choose Your Model", list(models))
    if user_model != "":
        k=None
        if user_model in ["K-Means","GMM","Agglomerative"]:
            k = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=3)
        preprocessor = model_dev.preprocess_data(df).fit_transform(df)
        trained_model = model_dev.fit_clusters(preprocessor, user_model, k)
        # Display predictions 
        st.subheader("Clustering Predictions:")
        df['Cluster'] = trained_model
        st.write(df)
        download_button = st.download_button(
        label="Download DataFrame with Clusters",
        key="download_clusters",
        data=df.to_csv(index=False),
        file_name="clusters_data.csv",
        mime="text/csv")

        metric = st.selectbox("Choose Your METRIC",['Davies Bouldin Score', 'Silhouette Score'])
        results = model_dev.evaluate_clusters(preprocessor, trained_model, metric)
        if metric== 'Silhouette Score':
            if results<0.7:
                st.warning('You may want to fine tune your model or use another one.')
            else:
                st.success('Nice performance!')
        #else:

        # Display model history
        st.subheader("Model History")
        session_state.clu_model_history[user_model] = {'metric': metric, 'result': results}
        for model, history in session_state.clu_model_history.items():
            st.write(f"{model}: Metric={history['metric']}, Result={history['result']}")
        
        # Display clusters 
        st.subheader("Cluster Visualization")
        model_dev.visualize_clusters(preprocessor, trained_model)