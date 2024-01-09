from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st 
import maps 
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score, adjusted_rand_score
from sklearn.model_selection import GridSearchCV, ParameterGrid

'''
Code for functions related to model preprocessing, training, evaluation, visualization.
Includes: 
    - Preprocessing pipeline creation function
    - Model fitting with pipeline and hyper parameter tuning function 
    - Model prediction output function 
    - Model evaluation using user input metric function
    - Feature importance visualization function 
    - Cluster fitting, evaluation, and visualization functions 
'''

def preprocess_data(X):
    """
    Preprocess the data by selecting numerical and categorical columns and creating a preprocessor.
    """
    # Separate numerical and categorical columns 
    num_columns = X.select_dtypes(include=['int', 'float']).columns.tolist()
    cat_columns = [col for col in X.columns if col not in num_columns and col not in text_columns]
    
    # Create preprocessing pipeline to transform columns into model input
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', Pipeline(steps=[
                ('scaler', StandardScaler())
            ]), num_columns),
            ('categorical', Pipeline(steps=[
                ('onehot', OneHotEncoder())
            ]), cat_columns)
        ], sparse_threshold=0)
    
    return preprocessor

def fit_model(X_train, y_train, preprocessor, user_model, param=False):
    """
    Fit the model on the training data using parameter tuning if selected.
    """

    # Get model function using name mapping 
    model = maps.sup_models[user_model][0]

    # Condition to differentiate between multiclassification and binary classification problems 
    num_classes = len(np.unique(y_train))

    # Preprocess x and y data 
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)])
    y_train_encoded = LabelEncoder().fit_transform(y_train)

    # If multiclass classification problem tune param for xgboost model 
    if num_classes>2 and user_model=='XGBoost (Classifier)':
        model.set_params(objective='multi:softprob', num_class=num_classes)
    
    # If user chooses to tune parameters extract mapping and apply grid search cv to get best model
    if param:
        param_grid = maps.sup_models[user_model][2]
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train_encoded)

        # Display optimal parameters 
        st.info(f"Optimal Hyper Parameters: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_

    # If user chooses not to tune parameters fit baseline model
    else:
        pipeline.fit(X_train, y_train_encoded)
        best_model = pipeline
    
    return best_model

def predict_model(model, X_test):
    """
    Predict using the trained model.
    """
    predictions = model.predict(X_test)
    return predictions

def evaluate_model(y_true, y_pred, metric, X, average='binary'):
    """
    Evaluate the model using user specified metric.
    """
    # Preprocess test data prior to comparison 
    y_true_encoded = LabelEncoder().fit_transform(y_true)

    if metric == "Adjusted R2":
        return maps.metrics[metric](y_true, y_pred, X)
    if metric in ['Precision', 'Recall', 'F1 Score']:
        return maps.metrics[metric](y_true_encoded, y_pred, average=average)
    else:
        return maps.metrics[metric](y_true_encoded, y_pred)

def feature_importances(model_name, trained_model, top_n=12):
    """
    Extract highly impactful features of model and visualize up to top 12. 
    """
    if model_name in ("K-Nearest Neighbors (Regressor)", "K-Nearest Neighbors (Classifier)"):
        st.write("No feature importances applicable for this model")
        return 
    
    # Extract appropriate coefficient attribute for model from mapping 
    attr = maps.sup_models[model_name][1]
    if model_name in ('Logistic Regression'):
        importances = getattr(trained_model.named_steps['model'], attr)[0]
    else:
        importances = getattr(trained_model.named_steps['model'], attr)
    
    # Extract and sort features from pipeline
    feature_names = trained_model.named_steps['preprocessor'].get_feature_names_out()
    indices = np.argsort(importances)[::-1]
    
    # Determine the length to avoid index out of range error
    plot_len = min(len(feature_names), top_n, len(indices))

    # Plot the top features
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(range(plot_len), importances[indices][:plot_len], align="center")
    ax.set_xticks(range(plot_len))
    ax.set_xticklabels([feature_names[i] for i in indices][:plot_len], rotation=45, ha="right")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance/Coefficient")
    ax.set_title(f"Top {plot_len} Features Importance/Coefficients")
    st.pyplot(fig)

def fit_clusters(df, model, k=None):
    """
    Fit clustering model using user input k if given 
    """
    # Extract model function and cluster attribute from mapping 
    model_class, param_name= maps.unsup_models[model]

    if param_name:
        model = model_class(**{param_name: k})
    else:
        model = model_class()

    # Return cluster labels 
    predictions = model.fit_predict(df)
    return predictions

def evaluate_clusters(df, predictions, metric):
    """
    Map clustering metrics to functions and evaluate on predictions
    """
    metrics = {'Davies Bouldin Score': davies_bouldin_score,
     'Silhouette Score':silhouette_score}
    return metrics[metric](df, predictions)

def visualize_clusters(df, predictions):
    """
    Apply PCA to reduce dimensionality of feature and visualize clusters 
    """
    # Clear previous plt plots 
    plt.clf()

    # Use PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(df)

    # Get unique cluster labels
    unique_clusters = list(set(predictions))

    # Create a scatter plot for each cluster
    for cluster_label in unique_clusters:
        cluster_indices = predictions == cluster_label
        plt.scatter(reduced_features[cluster_indices, 0], reduced_features[cluster_indices, 1],
                    label=f'Cluster {cluster_label}', alpha=0.5)

    plt.title("Clustering Results (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()

    st.pyplot()

    
