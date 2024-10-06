from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
import os
import pandas as pd
import numpy as np

def train_xgboost_model(X_train, y_train, X_test, y_test):
    # Convert to DataFrame if not already
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

    # Identify categorical columns (assuming categorical columns are object or category dtype)
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Encode categorical variables
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initial model training
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Hyperparameter tuning
    param_dist = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Score: {random_search.best_score_}")

    # Train optimized model
    optimized_model = XGBClassifier(**random_search.best_params_)
    optimized_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = optimized_model.predict(X_test)
    y_pred_prob = optimized_model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    model_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(optimized_model, os.path.join(model_dir, 'xgboost_model.joblib'))
    joblib.dump(encoder, os.path.join(model_dir, 'xgboost_encoder.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'xgboost_scaler.joblib'))

    return optimized_model, encoder, scaler

def load_xgboost_model():
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(model_dir, 'xgboost_model.joblib'))
    try:
        encoder = joblib.load(os.path.join(model_dir, 'xgboost_encoder.joblib'))
        print(f"Loaded encoder. Fitted categories: {encoder.categories_ if hasattr(encoder, 'categories_') else 'Not available'}")
    except:
        print("No encoder found. Proceeding without encoding.")
        encoder = None
    try:
        scaler = joblib.load(os.path.join(model_dir, 'xgboost_scaler.joblib'))
        print(f"Loaded scaler. Scale: {scaler.scale_}")
        print(f"Scaler mean: {scaler.mean_}")
    except:
        print("No scaler found. Proceeding without scaling.")
        scaler = None
    return model, encoder, scaler