import argparse
import os
import pandas as pd
import numpy as np
from scapy.all import rdpcap, sniff
from scapy.layers.inet import IP, TCP, UDP 
from scapy.all import get_if_list, conf, get_working_if
import pyfiglet
from sklearn.ensemble import VotingClassifier, IsolationForest
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from system_main.dataset.data_processing import load_and_preprocess_data, preprocess_live_data, load_preprocessor
from system_main.model.random_forest_model import train_model as train_rf_model, load_model as load_rf_model, predict_with_probabilities as rf_predict
from system_main.model.xgboost_model import train_xgboost_model, load_xgboost_model, predict_with_probabilities as xgb_predict
from system_main.feature_extraction import extract_features_from_packet, extract_features_from_pcap, preprocess_features
def show_system_name(mode, input_file=None, interface=None):
    ascii_banner = pyfiglet.figlet_format("Network Anomaly Detection")
    print(ascii_banner)
    print(f"Mode: {mode}")
    print(f"Input: {input_file or interface or 'N/A'}")
    print("-" * 50)


def list_network_interfaces():
    print("Available network interfaces:")
    for iface in get_if_list():
        try:
            ip = conf.ifaces[iface].ip
        except AttributeError:
            ip = "N/A"
        print(f"- {iface}: IP {ip}")
    
    print(f"\nCurrent default interface: {get_working_if()}")

def ensemble_predict(rf_model, xgb_model, X):
    rf_pred, rf_prob = rf_predict(rf_model, X)
    xgb_pred, xgb_prob = xgb_predict(xgb_model, X)
    
    # Combine probabilities
    ensemble_prob = (rf_prob + xgb_prob) / 2
    ensemble_pred = np.argmax(ensemble_prob, axis=1)
    
    return ensemble_pred, ensemble_prob

def analyze_preprocessed_data(features_df):
    print("\nPreprocessed Data Analysis:")
    print(f"Shape: {features_df.shape}")
    print("\nFeature Statistics:")
    print(features_df.describe())
    print("\nMissing Values:")
    print(features_df.isnull().sum())
    print("\nData Types:")
    print(features_df.dtypes)

def generate_report(traffic_data, total_packets, probabilities, predictions, attack_labels, args, features_df, model_type, selected_features):
    normal_traffic = traffic_data[traffic_data['predicted_attack'] == 'Normal']
    anomalies = traffic_data[traffic_data['predicted_attack'] != 'Normal']
    
    normal_percentage = (len(normal_traffic) / total_packets) * 100 if total_packets > 0 else 0
    anomaly_percentage = (len(anomalies) / total_packets) * 100 if total_packets > 0 else 0
    
    report = f"""Analysis Report:
----------------
Total packets analyzed: {total_packets}
Normal packets: {len(normal_traffic)} ({normal_percentage:.2f}%)
Anomalous packets: {len(anomalies)} ({anomaly_percentage:.2f}%)

Detected Attack Types:
"""
    if len(anomalies) > 0:
        for attack_type, count in anomalies['predicted_attack'].value_counts().items():
            report += f"{attack_type}: {count} ({count/len(anomalies)*100:.2f}% of anomalies)\n"
        
        report += "\nTop 5 anomalous packets:"
        top_5_anomalies = anomalies.nlargest(5, 'attack_probability')
        for i, (idx, row) in enumerate(top_5_anomalies.iterrows(), 1):
            report += f"\nRank {i}: Packet {idx + 1}, Attack: {row['predicted_attack']}, Probability: {row['attack_probability']:.4f}"
    else:
        report += "No anomalous packets detected."

    if args.detail:
        report += f"""

Detailed Report:
----------------
Model type: {model_type}
Selected features: {selected_features}
Shape of feature array: {features_df.shape}

Feature Importance:
"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for feature, importance in zip(selected_features, importances):
                report += f"{feature}: {importance:.4f}\n"
        else:
            report += "Feature importance not available for this model type.\n"

        report += f"""
Sample of anomalous traffic (first 5 packets):
{traffic_data[traffic_data['is_anomaly']].head().to_string() if len(traffic_data[traffic_data['is_anomaly']]) > 0 else "No anomalous traffic detected"}
"""

    return report

def detect_anomalies_pcap(pcap_file, model, preprocessor, selected_features, model_type, args):
    print(f"Analyzing PCAP file: {pcap_file}")
    features_df = extract_features_from_pcap(pcap_file)
    
    if features_df is None or features_df.empty:
        print("No features could be extracted. Unable to perform detection.")
        return None, None

    print(f"Extracted features: {features_df.columns.tolist()}")
    features_df = preprocess_features(features_df)
    
    analyze_preprocessed_data(features_df)
    
    features_df, available_features = preprocess_live_data(features_df, preprocessor, selected_features)
    
    print(f"Features used for prediction: {available_features}")
    print(f"Missing features: {set(selected_features) - set(available_features)}")

    if features_df.empty:
        print("No usable features for prediction. Unable to perform detection.")
        return None, None

    if model_type == 'rf':
        predictions, probabilities = rf_predict(model, features_df)
    elif model_type == 'xgboost':
        predictions, probabilities = xgb_predict(model, features_df)
    elif model_type == 'ensemble':
        predictions, probabilities = ensemble_predict(model['rf'], model['xgb'], features_df)

    # Anomaly detection using Elliptic Envelope
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    ee = EllipticEnvelope(contamination=0.1, random_state=42)
    anomaly_scores = ee.fit_predict(scaled_features)

    # Convert numeric predictions back to attack labels
    attack_labels = preprocessor['label_encoders']['attack_cat'].classes_
    predicted_attacks = [attack_labels[p] for p in predictions]
    
    features_df['predicted_attack'] = predicted_attacks
    features_df['attack_probability'] = probabilities.max(axis=1)
    features_df['anomaly_score'] = anomaly_scores

    # Apply threshold for anomaly detection
    anomaly_threshold = 0.9  # Adjust this value as needed
    features_df['is_anomaly'] = (features_df['attack_probability'] > anomaly_threshold) | (features_df['anomaly_score'] == -1)

    report = generate_report(features_df, len(features_df), probabilities, predictions, attack_labels, args, features_df, model_type, available_features)
    print(report)

    return features_df, probabilities

def detect_anomalies_live(interface, model, preprocessor, selected_features, model_type, args):
    print(f"Monitoring live traffic on interface: {interface}")
    
    traffic_data = []
    total_packets = 0
    error_count = 0

    attack_labels = preprocessor['label_encoders']['attack_cat'].classes_

    iso_forest = IsolationForest(contamination=0.1, random_state=42)

    def packet_callback(packet):
        nonlocal total_packets, traffic_data, error_count
        total_packets += 1
        
        try:
            if IP in packet:
                features = extract_features_from_packet(packet, packet.time)
                if features:
                    features_df = pd.DataFrame([features])
                    features_df = preprocess_features(features_df)
                    features_df, _ = preprocess_live_data(features_df, preprocessor, selected_features)
        
                    if model_type == 'rf':
                        predictions, probabilities = rf_predict(model, features_df)
                    elif model_type == 'xgboost':
                        predictions, probabilities = xgb_predict(model, features_df)
                    elif model_type == 'ensemble':
                        predictions, probabilities = ensemble_predict(model['rf'], model['xgb'], features_df)
                    
                    anomaly_score = iso_forest.predict(features_df)
                    
                    predicted_attack = attack_labels[predictions[0]]
                    attack_prob = probabilities[0].max()
                    
                    traffic_data.append({**features, 'predicted_attack': predicted_attack, 'attack_probability': attack_prob, 'anomaly_score': anomaly_score[0]})
                    
                    if predicted_attack != 'Normal' or anomaly_score[0] == -1:
                        print(f"Anomaly detected: {packet.summary()}")
                        print(f"Predicted Attack: {predicted_attack}")
                        print(f"Probability: {attack_prob:.4f}")
                        print(f"Anomaly Score: {anomaly_score[0]}")
                    else:
                        print(f"Normal traffic: {packet.summary()}")
                        print(f"Normal Probability: {probabilities[0, attack_labels.tolist().index('Normal')]:.4f}")

            if total_packets % 100 == 0:
                report = generate_report(pd.DataFrame(traffic_data), total_packets, 
                                         np.array([t['attack_probability'] for t in traffic_data]),
                                         [attack_labels.tolist().index(t['predicted_attack']) for t in traffic_data],
                                         attack_labels, args, pd.DataFrame(traffic_data), model, selected_features)
                print(report)

        except Exception as e:
            error_count += 1
            print(f"Error processing packet: {e}")
            if error_count > 10:
                print("Too many errors. Stopping packet capture.")
                return True
            
    try:
        sniff(iface=interface, prn=packet_callback, store=0, stop_filter=lambda _: error_count > 10)
    except KeyboardInterrupt:
        print("\nStopped packet capture.")
    finally:
        print(f"Total packets analyzed: {total_packets}")
        print(f"Total anomalies detected: {len([t for t in traffic_data if t['predicted_attack'] != 'Normal' or t['anomaly_score'] == -1])}")
        print(f"Total errors encountered: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network Anomaly Detection System")
    parser.add_argument("--mode", choices=["train", "detect_pcap", "detect_live", "list_interfaces"], required=True)
    parser.add_argument("--model", choices=["rf", "xgboost", "ensemble"], default="rf")
    parser.add_argument("--train_file", default="UNSW_NB15_training-set.csv")
    parser.add_argument("--test_file", default="UNSW_NB15_testing-set.csv")
    parser.add_argument("--input", help="Input PCAP file for detection")
    parser.add_argument("--interface", help="Network interface for live detection")
    parser.add_argument("--detail", action="store_true", help="Generate detailed report")
    args = parser.parse_args()

    if args.mode == "list_interfaces":
        list_network_interfaces()
    else:
        show_system_name(args.mode, args.input, args.interface)

        if args.mode == "train":
            X_train, X_test, y_train, y_test, label_encoders, selected_features = load_and_preprocess_data(args.train_file, args.test_file)
            if args.model == "rf":
                model = train_rf_model(X_train, y_train, X_test, y_test, selected_features)
            elif args.model == "xgboost":
                model = train_xgboost_model(X_train, y_train, X_test, y_test, selected_features)
            elif args.model == "ensemble":
                rf_model = train_rf_model(X_train, y_train, X_test, y_test, selected_features)
                xgb_model = train_xgboost_model(X_train, y_train, X_test, y_test, selected_features)
                model = {
                    'rf': rf_model,
                    'xgb': xgb_model
                }
            print(f"Training completed. Model(s) saved.")
    
        elif args.mode in ["detect_pcap", "detect_live"]:
            preprocessor = load_preprocessor()
            if args.model == "rf":
                model = load_rf_model()
            elif args.model == "xgboost":
                model = load_xgboost_model()
            elif args.model == "ensemble":
                model = {
                    'rf': load_rf_model(),
                    'xgb': load_xgboost_model()
                }
            
            _, _, _, _, label_encoders, selected_features = load_and_preprocess_data(args.train_file, args.test_file)

            if args.mode == "detect_pcap":
                if not args.input:
                    print("Please provide an input PCAP file for detection.")
                    exit(1)
                detect_anomalies_pcap(args.input, model, preprocessor, selected_features, args.model, args)
            elif args.mode == "detect_live":
                if not args.interface:
                    print("Please provide a network interface for live detection.")
                    print("You can use the --mode list_interfaces option to see available interfaces.")
                    exit(1)
                detect_anomalies_live(args.interface, model, preprocessor, selected_features, args.model, args)