import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

def load_and_preprocess_data(train_file, test_file):
    # Identify the correct project root
    current_dir = os.getcwd()
    if 'system_main' in current_dir:
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    train_data_path = os.path.join(project_root, 'system_main', 'dataset', train_file)
    test_data_path = os.path.join(project_root, 'system_main', 'dataset', test_file)

    # Load the data
    train_data_initial = pd.read_csv(train_data_path)
    test_data_initial = pd.read_csv(test_data_path)

    # Drop the 'service' column
    train_data = train_data_initial.drop(columns=['service'])
    test_data = test_data_initial.drop(columns=['service'])

    # Define the target column and categorical columns
    target_column = 'label'
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
        X_train[col] = label_encoders[col].fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=20)
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_feature_names = X_train.columns[selector.get_support()]
    X_test_selected = X_test[selected_feature_names]

    return X_train_selected, X_test_selected, y_train, y_test, label_encoders, selected_feature_names