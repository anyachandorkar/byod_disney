from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc, f1_score
models = {
        "Random Forest (Classifier)": RandomForestClassifier(),
        "XGBoost (Classifier)": XGBClassifier(),
        "SVM (Classifier)": SVC(),
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors (Classifier)": KNeighborsClassifier(),
        "Random Forest (Regressor)": RandomForestRegressor(),
        "XGBoost (Regressor)": XGBRegressor(),
        "SVM (Regressor)": SVR(),
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors (Regressor)": KNeighborsRegressor(),
    }
metrics = {'Confusion Matrix': confusion_matrix,
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'ROC Curve': roc_curve,
    'Precision-Recall Curve': precision_recall_curve,
    'AUC (ROC)': auc,
    'AUC (Precision-Recall)': auc,
    'F1 Score': f1_score}

def preprocess_data(X):
    """
    Preprocess the data by selecting numerical and categorical columns and creating a preprocessor.
    """
    num_columns = X.select_dtypes(include=['int', 'float']).columns.tolist()
    cat_columns = [col for col in X.columns if col not in num_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', Pipeline(steps=[
                ('onehot', OneHotEncoder())
            ]), cat_columns),
            ('numeric', Pipeline(steps=[
                ('scaler', StandardScaler())
            ]), num_columns)
        ])
    return num_columns, cat_columns, preprocessor

def fit_model(X_train, y_train, preprocessor, user_model):
    """
    Fit the model on the training data.
    """
    model = models[user_model]
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)])
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_model(model, X_test):
    """
    Predict using the trained model.
    """
    predictions = model.predict(X_test)
    return predictions

def evaluate_model(y_true, y_pred, metric):
    """
    Evaluate the model using a specific metric.
    """
    return metrics[metric](y_true, y_pred)

    

    
