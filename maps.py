from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np 

'''
Code for Dictionary mapping 

Includes: 
    - Metric name to metric function
    - Param grids for all models 
    - Model name to model function, coefficient extraction function, and param grids
    for both supervised and unsupervised models 
'''
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

param_grid_rf = {
    'model__n_estimators': [10, 100, 1000],
    'model__max_features': ['sqrt', 'log2'],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}
param_grid_xgb = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 6, 9],
    'model__subsample': [0.8, 1.0]
}
param_grid_lgr = {
    'model__C': [0.001, 0.01, 0.1, 1, 10],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear']
}
param_grid_knn = {
    'model__n_neighbors': [3, 5, 7],
    'model__weights': ['uniform', 'distance'],
    'model__p': [1, 2]
}

param_grid_kmeans = {
        'estimator__init': ['k-means++', 'random'],
        'estimator__max_iter': [100, 300, 500]
    }
param_grid_dbs = {
        'estimator__eps': [0.1, 0.3, 0.5],
        'estimator__min_samples': [3, 5, 7],
    }
param_grid_gmm = {
        'model__n_components': [2, 3, 4],
        'model__covariance_type': ['full', 'tied', 'diag', 'spherical'],
    }
param_grid_agg = {
        'model__n_clusters': [2, 3, 4, 5],
        'model__linkage': ['ward', 'complete', 'average', 'single'],
    }

# supervised model map 
sup_models = {
        "Random Forest (Classifier)": [RandomForestClassifier(), 'feature_importances_', param_grid_rf],
        "XGBoost (Classifier)": [XGBClassifier(), 'feature_importances_', param_grid_xgb],
        "Logistic Regression": [LogisticRegression(),'coef_', param_grid_lgr],
        "K-Nearest Neighbors (Classifier)": [KNeighborsClassifier(), '', param_grid_knn],
        "Random Forest (Regressor)": [RandomForestRegressor(), 'feature_importances_', param_grid_rf],
        "XGBoost (Regressor)": [XGBRegressor(),'feature_importances_', param_grid_xgb],
        "Linear Regression": [LinearRegression(),'coef_', {}],
        "K-Nearest Neighbors (Regressor)": [KNeighborsRegressor(), '', param_grid_knn]
    }

#unsupervised model map 
unsup_models = {
    "K-Means": [KMeans, "n_clusters"],
    "DBSCAN": [DBSCAN, None],
    "GMM": [GaussianMixture, "n_components"],
    "Agglomerative": [AgglomerativeClustering, "n_clusters"]}
