from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np 

# supervised model map 
sup_models = {
        "Random Forest (Classifier)": [RandomForestClassifier(), 'feature_importances_'],
        "XGBoost (Classifier)": [XGBClassifier(), 'feature_importances_'],
        "Logistic Regression": [LogisticRegression(),'coef_'],
        "K-Nearest Neighbors (Classifier)": [KNeighborsClassifier()],
        "Random Forest (Regressor)": [RandomForestRegressor(), 'feature_importances_'],
        "XGBoost (Regressor)": [XGBRegressor(),'feature_importances_'],
        "Linear Regression": [LinearRegression(),'coef_'],
        "K-Nearest Neighbors (Regressor)": [KNeighborsRegressor()]
    }

#unsupervised model map 
unsup_models = {
    "K-Means": (KMeans, "n_clusters"),
    "DBSCAN": (DBSCAN, None),
    "GMM": (GaussianMixture, "n_components"),
    "Agglomerative": (AgglomerativeClustering, "n_clusters")}

# evaluation metric map 
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
