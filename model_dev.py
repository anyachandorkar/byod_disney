from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st 
from sklearn.base import BaseEstimator, TransformerMixin
models = {
        "Random Forest (Classifier)": [RandomForestClassifier(), 'feature_importances_'],
        "XGBoost (Classifier)": [XGBClassifier(), 'feature_importances_'],
        "Logistic Regression": [LogisticRegression(),'coef_'],
        "K-Nearest Neighbors (Classifier)": [KNeighborsClassifier()],
        "Random Forest (Regressor)": [RandomForestRegressor(), 'feature_importances_'],
        "XGBoost (Regressor)": [XGBRegressor(),'feature_importances_'],
        "Linear Regression": [LinearRegression(),'coef_'],
        "K-Nearest Neighbors (Regressor)": [KNeighborsRegressor()]
    }
metrics = {'Confusion Matrix': confusion_matrix,
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1 Score': f1_score,
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
    'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'R2': r2_score,
    'Adjusted R2': lambda y_true, y_pred, X: 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1)}

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
    
def preprocess_data(X, text_columns):
    """
    Preprocess the data by selecting numerical and categorical columns and creating a preprocessor.
    """
    num_columns = X.select_dtypes(include=['int', 'float']).columns.tolist()
    cat_columns = [col for col in X.columns if col not in num_columns and col not in text_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', Pipeline(steps=[
                ('scaler', StandardScaler())
            ]), num_columns),
            ('categorical', Pipeline(steps=[
                ('onehot', OneHotEncoder())
            ]), cat_columns),
            ('text', Pipeline(steps=[
                ('tfidf', TfidfVectorizer())
            ]), text_columns)
        ], sparse_threshold=0)
    
    return preprocessor

def fit_model(X_train, y_train, preprocessor, user_model):
    """
    Fit the model on the training data.
    """
    model = models[user_model][0]
    num_classes = len(np.unique(y_train))
    y_train_encoded = LabelEncoder().fit_transform(y_train)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)])
    if num_classes>2 and user_model=='XGBoost (Classifier)':
        model.set_params(objective='multi:softprob', num_class=num_classes)
    pipeline.fit(X_train, y_train_encoded)
    return pipeline

def predict_model(model, X_test):
    """
    Predict using the trained model.
    """
    predictions = model.predict(X_test)
    return predictions

def evaluate_model(y_true, y_pred, metric, X, average='binary'):
    """
    Evaluate the model using a specific metric.
    """
    y_true_encoded = LabelEncoder().fit_transform(y_true)
    if metric == "Adjusted R2":
        return metrics[metric](y_true, y_pred, X)
    if metric in ['Precision', 'Recall', 'F1 Score']:
        return metrics[metric](y_true_encoded, y_pred, average=average)
    else:
        return metrics[metric](y_true_encoded, y_pred)

def feature_importances(model_name, trained_model, top_n=12):
    if model_name in ("K-Nearest Neighbors (Regressor)", "K-Nearest Neighbors (Classifier)"):
        st.write("No feature importances applicable for this model")
        return 
    attr = models[model_name][1]
    if model_name in ('Logistic Regression'):
        importances = getattr(trained_model.named_steps['model'], attr)[0]
    else:
        importances = getattr(trained_model.named_steps['model'], attr)
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


    

    
