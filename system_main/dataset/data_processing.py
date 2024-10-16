import pandas as pd
import numpy as np
import joblib
import os

def engineer_features(df):
    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        df['total_bytes'] = df['sbytes'] + df['dbytes']
        df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1)

    if 'sttl' in df.columns and 'dttl' in df.columns:
        df['ttl_diff'] = df['sttl'] - df['dttl']

    if 'sport' in df.columns and 'dport' in df.columns:
        df['high_port'] = ((df['sport'] > 1024) | (df['dport'] > 1024)).astype(int)

    return df

def preprocess_for_inference(df, preprocessor, model_type):
    df = engineer_features(df)

    if model_type == 'rf':
        return preprocess_for_random_forest(df, preprocessor)
    elif model_type == 'xgboost':
        return preprocess_for_xgboost(df, preprocessor)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def preprocess_for_random_forest(df, preprocessor):
    for col, encoder in preprocessor['label_encoders'].items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    selected_features = preprocessor['selected_features']
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0

    return df[selected_features]

def preprocess_for_xgboost(df, preprocessor):
    categorical_columns = preprocessor['categorical_columns']
    numeric_columns = preprocessor['numeric_columns']
    encoder = preprocessor['encoder']
    expected_feature_order = preprocessor.get('feature_names', ['id', 'dur', 'proto', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'total_bytes', 'byte_ratio', 'ttl_diff'])

    # Create a new DataFrame with the expected columns
    processed_df = pd.DataFrame(index=df.index, columns=expected_feature_order)

    # Handle all columns
    for col in expected_feature_order:
        if col in df.columns:
            if col in categorical_columns:
                known_categories = encoder.categories_[encoder.feature_names_in_.tolist().index(col)]
                category_map = {cat: i for i, cat in enumerate(known_categories)}
                processed_df[col] = df[col].map(lambda x: category_map.get(x, len(category_map)))
            else:
                processed_df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            processed_df[col] = 0  # or another appropriate default value

    # Fill NaN values
    for col in processed_df.columns:
        median_value = processed_df[col].median()
        processed_df[col] = processed_df[col].fillna(median_value if pd.notnull(median_value) else -1)

    # Ensure all columns are numeric
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        processed_df[col] = processed_df[col].fillna(processed_df[col].median() if pd.notnull(processed_df[col].median()) else -1)

    return processed_df
def load_preprocessor(model_type):
    model_dir = os.path.join(os.getcwd(), 'system_main', 'model', 'saved_model')
    preprocessor_path = os.path.join(model_dir, f'{model_type}_preprocessor.joblib')
    return joblib.load(preprocessor_path)