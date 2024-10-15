import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib

def improved_feature_selection(X, y):
    # Remove features with low variance
    var_threshold = VarianceThreshold(threshold=0.01)
    X_var = var_threshold.fit_transform(X)
    
    # Select features based on mutual information
    mi_scores = mutual_info_classif(X_var, y, random_state=42)
    mi_threshold = np.percentile(mi_scores, 50)  # Select top 50% features
    selected_features = mi_scores >= mi_threshold
    
    # Get the names of selected features
    selected_feature_names = X.columns[var_threshold.get_support()][selected_features]
    
    return X.loc[:, selected_feature_names], selected_feature_names

def balance_classes(X, y):
    oversample = SMOTE(sampling_strategy='auto', random_state=42)
    undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    steps = [('o', oversample), ('u', undersample)]
    pipeline = Pipeline(steps=steps)
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    return X_resampled, y_resampled

def load_and_preprocess_data(train_file, test_file, sample_size=100000):
    # Identify the correct project root
    current_dir = os.getcwd()
    if 'system_main' in current_dir:
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    train_data_path = os.path.join(project_root, 'system_main', 'dataset', train_file)
    test_data_path = os.path.join(project_root, 'system_main', 'dataset', test_file)

     # Load the data with sampling
    train_data = pd.read_csv(train_data_path).sample(n=min(sample_size, len(pd.read_csv(train_data_path))), random_state=42)
    test_data = pd.read_csv(test_data_path).sample(n=min(sample_size // 2, len(pd.read_csv(test_data_path))), random_state=42)
    # Drop the 'service' column
    train_data = train_data.drop(columns=['service'])
    test_data = test_data.drop(columns=['service'])

    # Define the target column and categorical columns
    # Define the target column and categorical columns
    target_column = 'attack_cat'
    categorical_columns = ['proto', 'state', 'attack_cat']

    # Separate features and target
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Encode categorical columns
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        if col in X_train.columns:
            X_train[col] = label_encoders[col].fit_transform(X_train[col].astype(str))
            X_test[col] = X_test[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)
        elif col == target_column:
            y_train = label_encoders[col].fit_transform(y_train.astype(str))
            y_test = y_test.apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Improved feature selection
    X_train, selected_features = improved_feature_selection(X_train, y_train)
    X_test = X_test.loc[:, X_train.columns]  # Select the same features for test set

    # Balance classes for training data
    X_train, y_train = balance_classes(X_train, y_train)

    # Create preprocessor dictionary
    preprocessor = {
        'label_encoders': label_encoders,
        'imputer': imputer,
        'selected_features': X_train.columns.tolist()
    }

    # Save preprocessor
    preprocessor_path = os.path.join(project_root, 'preprocessor.joblib')
    try:
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")
    except Exception as e:
        print(f"Error saving preprocessor: {e}")

    return X_train, X_test, y_train, y_test, label_encoders, X_train.columns.tolist()

def preprocess_live_data(df, preprocessor, selected_features):
    # Only use features that are both in the DataFrame and in selected_features
    available_features = list(set(df.columns) & set(selected_features))
    df = df[available_features].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Encode categorical columns
    for col, encoder in preprocessor['label_encoders'].items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ else -1)

    # Handle missing features
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0 

    # Ensure all columns are in the same order as during training
    df = df.reindex(columns=selected_features, fill_value=0)

    # We'll skip imputation for now
    
    return df, available_features
def load_preprocessor():
    current_dir = os.getcwd()
    if 'system_main' in current_dir:
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir
    preprocessor_path = os.path.join(project_root, 'preprocessor.joblib')
    try:
        return joblib.load(preprocessor_path)
    except FileNotFoundError:
        print(f"Error: Preprocessor file not found at {preprocessor_path}")
        raise
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        raise