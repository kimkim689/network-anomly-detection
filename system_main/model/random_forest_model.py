from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import numpy as np
import pandas as pd

def train_model(X_train, y_train, X_test, y_test, feature_names=None):
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1, scoring='f1_macro')
    random_search.fit(X_train, y_train)
    
    model = random_search.best_estimator_

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    # Generate and print classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # Generate and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Feature importance
    if feature_names is not None:
        importances = model.feature_importances_
        forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print("\nFeature Importances:")
        print(forest_importances)

    # Save the model
    model_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(model_dir, 'random_forest_model.joblib'))

    return model

def load_model():
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(model_dir, 'random_forest_model.joblib'))
    return model

def predict_with_probabilities(model, X):
    return model.predict(X), model.predict_proba(X)