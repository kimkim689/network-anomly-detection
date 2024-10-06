from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import pandas as pd
import numpy as np

def train_model(X_train, y_train, X_test, y_test, feature_names=None):
    # Ensure X_train and X_test are pandas DataFrames
    if not isinstance(X_train, pd.DataFrame):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")
    
    # Generate and print classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    # Save the model
    model_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(model_dir, 'random_forest_model.joblib'))
    
    # Save feature names
    joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, 'feature_names.joblib'))
    
    return model

def load_model():
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(model_dir, 'random_forest_model.joblib'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
    return model, feature_names