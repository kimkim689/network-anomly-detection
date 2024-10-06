import argparse
import os
import pandas as pd
import numpy as np
from scapy.all import rdpcap, sniff
import pyfiglet
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from system_main.dataset.data_processing import load_and_preprocess_data
from system_main.model.random_forest_model import train_model as train_rf_model, load_model as load_rf_model
from system_main.model.xgboost_model import train_xgboost_model, load_xgboost_model
from system_main.feature_extraction import extract_features_from_packet, extract_features_from_pcap, preprocess_features


def show_system_name(mode, input_file=None, interface=None):
    ascii_banner = pyfiglet.figlet_format("Network Anomaly Detection")
    print(ascii_banner)
    print(f"Mode: {mode}")
    print(f"Input: {input_file or interface or 'N/A'}")
    print("-" * 50)

def generate_report(anomalies, total_packets):
    anomaly_percentage = (len(anomalies) / total_packets) * 100 if total_packets > 0 else 0
    report = f"""
    Analysis Report:
    ----------------
    Total packets analyzed: {total_packets}
    Anomalies detected: {len(anomalies)}
    Percentage of anomalies: {anomaly_percentage:.2f}%

    Recommendation:
    """
    if anomaly_percentage < 1:
        report += "Low level of anomalies detected. Continue monitoring."
    elif 1 <= anomaly_percentage < 5:
        report += "Moderate level of anomalies detected. Investigate the anomalous packets."
    else:
        report += "High level of anomalies detected. Urgent investigation required."
    
    return report

def detect_anomalies_pcap(pcap_file, model, encoder, scaler, selected_features, model_type):
    print(f"Analyzing PCAP file: {pcap_file}")
    features_df = extract_features_from_pcap(pcap_file)
    features_df = preprocess_features(features_df)
    
    print(f"Features in PCAP data: {features_df.columns.tolist()}")
    
    if model_type == 'xgboost':
        # For XGBoost, use the first 20 features (as that's what the model expects)
        selected_features = features_df.columns[:20].tolist()
    elif selected_features is None:
        selected_features = features_df.columns.tolist()
    
    print(f"Selected features: {selected_features}")

    # Add missing features with default values for Random Forest
    if model_type == 'rf':
        missing_features = [f for f in selected_features if f not in features_df.columns]
        for feature in missing_features:
            features_df[feature] = 0  # or another appropriate default value
        print(f"Added missing features with default values: {missing_features}")

    # Use only selected features
    features_df = features_df[selected_features]
    
    categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = features_df.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"Categorical columns: {categorical_cols.tolist()}")
    print(f"Numerical columns: {numerical_cols.tolist()}")

    # Handle categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        features_df[col] = le.fit_transform(features_df[col].astype(str))

    # Fill missing values with 0
    features_df = features_df.fillna(0)

    if model_type == 'xgboost' and scaler is not None:
        # Scale all features for XGBoost
        features_df = pd.DataFrame(scaler.transform(features_df), columns=features_df.columns)
        print(f"Scaled features: {features_df.columns.tolist()}")

    features_array = features_df.values

    print(f"Shape of features array: {features_array.shape}")

    # Get probability scores from the model
    probabilities = model.predict_proba(features_array)
    
    print("Anomaly probability scores:")
    for i, prob in enumerate(probabilities[:, 1]):
        print(f"Packet {i+1}: {prob:.4f}")


    # Use IsolationForest as a secondary detection method
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest_predictions = iso_forest.fit_predict(features_array)
    iso_forest_anomalies = iso_forest_predictions == -1

    # Combine predictions
    anomaly_threshold = np.percentile(probabilities[:, 1], 90)  # Flag top 10% as anomalies
    combined_anomalies = np.logical_or(probabilities[:, 1] > anomaly_threshold, iso_forest_anomalies)

    anomalies = features_df[combined_anomalies]

    report = generate_report(anomalies, len(features_df))
    print("\nAnalysis Report:")
    print(report)

    if len(anomalies) == 0:
        print("No anomalies detected. Consider reviewing the model and detection method.")
    else:
        print(f"Detected {len(anomalies)} anomalies. Review the detailed packet information above.")

    return anomalies, probabilities[:, 1]

def detect_anomalies_live(interface, model, encoder, scaler, selected_features, model_type, feature_names=None):
    print(f"Monitoring live traffic on interface: {interface}")
    
    anomalies = []
    total_packets = 0
    buffer = []

    def packet_callback(packet):
        nonlocal total_packets, anomalies, buffer
        total_packets += 1
        features = extract_features_from_packet(packet)
        if features:
            features_df = pd.DataFrame([features])
            features_df = preprocess_features(features_df)
            
            if model_type == 'xgboost':
                # Apply encoding and scaling for XGBoost
                categorical_cols = ['proto', 'service', 'state', 'attack_cat']
                features_df[categorical_cols] = encoder.transform(features_df[categorical_cols])
                features_df = scaler.transform(features_df)
            elif model_type == 'cnn':
                # Preprocess for CNN
                features_df = scaler.transform(features_df)
                buffer.append(features_df[0])
                if len(buffer) < 10:  # Assuming time_steps=10 for CNN
                    return
                features_df = np.array([buffer[-10:]])
            else:  # Random Forest
                for col in encoder:
                    if col in features_df.columns:
                        features_df[col] = encoder[col].transform(features_df[col].astype(str))
            
            # Ensure all selected features are present and in the correct order
            for feature in feature_names:
                if feature not in features_df.columns:
                    features_df[feature] = 0  # or another appropriate default value
            features_df = features_df[feature_names]
            # Ensure all selected features are present
            if model_type != 'cnn':
                for feature in selected_features:
                    if feature not in features_df.columns:
                        features_df[feature] = 0  # or another appropriate default value
                features_df = features_df[selected_features]
            
            prediction = model.predict(features_df)[0]
            if model_type == 'cnn':
                prediction = (prediction > 0.5).astype(int)
            if prediction == 1:
                print(f"Anomaly detected: {packet.summary()}")
                print(f"Features: {features}")
                anomalies.append(features)

        if total_packets % 100 == 0:  # Generate report every 100 packets
            report = generate_report(anomalies, total_packets)
            print(report)

    sniff(iface=interface, prn=packet_callback, store=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNSW_NB15 Network Anomaly Detection System")
    parser.add_argument("--mode", choices=["train", "detect_pcap", "detect_live"], required=True)
    parser.add_argument("--model", choices=["rf", "xgboost", "cnn"], default="rf", help="Choose model: Random Forest (rf), XGBoost (xgboost), or CNN (cnn)")
    parser.add_argument("--train_file", default="UNSW_NB15_training-set.csv", help="Training data CSV file")
    parser.add_argument("--test_file", default="UNSW_NB15_testing-set.csv", help="Testing data CSV file")
    parser.add_argument("--input", help="Input PCAP file for detection")
    parser.add_argument("--interface", help="Network interface for live detection")

    args = parser.parse_args()

    show_system_name(args.mode, args.input, args.interface)

    if args.mode == "train":
        X_train, X_test, y_train, y_test, label_encoders, selected_features = load_and_preprocess_data(args.train_file, args.test_file)
        if args.model == "rf":
            # Ensure selected_features is a list
            if isinstance(selected_features, pd.Index):
                selected_features = selected_features.tolist()
            elif selected_features is None:
                selected_features = [f"feature_{i}" for i in range(X_train.shape[1])]

            model = train_rf_model(X_train, y_train, X_test, y_test, selected_features)
            encoder, scaler = label_encoders, None
        elif args.model == "xgboost":
            model, encoder, scaler = train_xgboost_model(X_train, y_train, X_test, y_test)
        else:  # CNN
            model, scaler = train_cnn_model(X_train, y_train, X_test, y_test)
            encoder = None
    
    elif args.mode in ["detect_pcap", "detect_live"]:
        if args.model == "rf":
            model, feature_names = load_rf_model()
            _, _, _, _, encoder, selected_features = load_and_preprocess_data(args.train_file, args.test_file)
            scaler = None
        elif args.model == "xgboost":
            model, encoder, scaler = load_xgboost_model()
            _, _, _, _, _, selected_features = load_and_preprocess_data(args.train_file, args.test_file)
            feature_names = None
        else:  # CNN
            model, scaler = load_cnn_model()
            encoder, feature_names = None, None
            _, _, _, _, _, selected_features = load_and_preprocess_data(args.train_file, args.test_file)
        

    if args.mode == "detect_pcap":
        if args.model == "rf":
            model, selected_features = load_rf_model()
            encoder, scaler = None, None
        elif args.model == "xgboost":
            model, encoder, scaler = load_xgboost_model()
            if hasattr(model, 'feature_names_in_'):
                selected_features = model.feature_names_in_.tolist()
            else:
                selected_features = None
                print("Warning: No selected features found for XGBoost model. Using all available features.")
            if scaler is not None:
                if hasattr(scaler, 'feature_names_in_'):
                    print(f"Scaler features: {scaler.feature_names_in_.tolist()}")
                else:
                    print("Scaler does not have feature_names_in_ attribute. Assuming it was fitted on the first 20 features.")
    
        else:  # CNN
            model, scaler = load_cnn_model()
            encoder, selected_features = None, None
        
        print(f"Model type: {type(model)}")
        print(f"Selected features: {selected_features}")
        
        anomalies, probabilities = detect_anomalies_pcap(args.input, model, encoder, scaler, selected_features, args.model)
        
        if len(anomalies) > 0:
            print("\nDetailed information about detected anomalies:")
            print(anomalies)
        
        print("\nTop 10 packets with highest anomaly scores:")
        top_10_indices = np.argsort(probabilities)[-10:][::-1]
        for i, idx in enumerate(top_10_indices, 1):
            print(f"Rank {i}: Packet {idx + 1}, Score: {probabilities[idx]:.4f}")