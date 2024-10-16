import argparse
import os
import pandas as pd
import numpy as np
import joblib
from scapy.all import rdpcap, sniff
from scapy.layers.inet import IP
from scapy.all import get_if_list, conf, get_working_if
import pyfiglet
from sklearn.ensemble import IsolationForest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from system_main.dataset.data_processing import load_preprocessor, preprocess_for_inference, engineer_features
from system_main.feature_extraction import extract_features_from_packet, extract_features_from_pcap
import logging

DEFAULT_PERCENTILE = 90
def execute_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    return nb

def train_and_evaluate_models(args):
    model_dir = os.path.join(os.getcwd(), 'system_main', 'model', 'saved_model')
    
    if args.model in ['rf', None]:
        print("Training Random Forest model...")
        rf_notebook_path = os.path.join(model_dir, 'rfmodel.ipynb')
        rf_nb = execute_notebook(rf_notebook_path)
        print_model_evaluation(rf_nb, "Random Forest")
    
    if args.model in ['xgboost', None]:
        print("\nTraining XGBoost model...")
        xgb_notebook_path = os.path.join(model_dir, 'xgboostmodel.ipynb')
        xgb_nb = execute_notebook(xgb_notebook_path)
        print_model_evaluation(xgb_nb, "XGBoost")
    
    print("\nModel training completed.")
def print_model_evaluation(notebook, model_name):
    logging.info(f"Extracting evaluation metrics for {model_name} model...")
    evaluation_found = False
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            for output in cell.outputs:
                if output.output_type == 'stream':
                    if 'Optimized Model Accuracy:' in output.text:
                        print(f"\n{model_name} Model Evaluation:")
                        print(output.text)
                        evaluation_found = True
                        break
        if evaluation_found:
            break
    
    if not evaluation_found:
        logging.warning(f"No evaluation metrics found for {model_name} model in the notebook output.")
    logging.info(f"{model_name} model training and evaluation completed.")


def load_model(model_type):
    model_dir = os.path.join(os.getcwd(), 'system_main', 'model', 'saved_model')
    if model_type == 'rf':
        model_path = os.path.join(model_dir, 'random_forest_model.joblib')
    elif model_type == 'xgboost':
        model_path = os.path.join(model_dir, 'xgboost_model.joblib')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return joblib.load(model_path)


def combine_predictions(rf_prob, xgb_prob, threshold=0.7):
    combined_prob = np.zeros_like(rf_prob)
    combined_prob[rf_prob > threshold] = rf_prob[rf_prob > threshold]
    combined_prob[xgb_prob <= threshold] = xgb_prob[xgb_prob <= threshold]
    return combined_prob

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

def generate_report(anomalies, total_packets, probabilities, args, features_df, model_type, threshold):
    anomaly_percentage = (len(anomalies) / total_packets) * 100 if total_packets > 0 else 0
    
    report = f"""
Network Anomaly Detection Report
--------------------------------
1. Summary:
   - Total packets: {total_packets}
   - Anomalies detected: {len(anomalies)} ({anomaly_percentage:.2f}%)
   - Anomaly threshold: {threshold:.4f} (based on {DEFAULT_PERCENTILE}th percentile)

2. Probability Statistics:
   - Mean: {np.mean(probabilities):.4f}
   - Median: {np.median(probabilities):.4f}
   - Min: {np.min(probabilities):.4f}
   - Max: {np.max(probabilities):.4f}

3. Top 3 Anomalous Packets:
"""
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    for i, idx in enumerate(top_3_indices, 1):
        report += f"   {i}. Packet {idx + 1}, Score: {probabilities[idx]:.4f}\n"

    report += f"""
4. Recommendation:
   {get_recommendation(anomaly_percentage)}
"""

    if args.detail:
        report += f"""

Detailed Information:
---------------------
5. PCAP Features:
   {features_df.columns.tolist()}

7. Probability Scores (Top 10 and Bottom 10):
{get_probability_scores(probabilities, features_df)}
"""

    return report

def get_probability_scores(probabilities, features_df):
    # Create a dataframe with probabilities and basic packet info
    prob_df = pd.DataFrame({
        'Packet': range(1, len(probabilities) + 1),
        'Probability': probabilities,
    })

    # Sort by probability
    prob_df_sorted = prob_df.sort_values('Probability', ascending=False)

    # Get top 10 and bottom 10
    top_10 = prob_df_sorted.head(10)
    bottom_10 = prob_df_sorted.tail(10)

    # Format the output
    output = "   Top 10 packets by anomaly score:\n"
    output += top_10.to_string(index=False) + "\n\n"
    output += "   Bottom 10 packets by anomaly score:\n"
    output += bottom_10.to_string(index=False)

    return output
def get_recommendation(anomaly_percentage):
    if anomaly_percentage < 10:
        return "Low level of anomalies. Continue monitoring."
    elif 10 <= anomaly_percentage < 20:
        return "Moderate level of anomalies. Investigate top anomalous packets."
    elif 20 <= anomaly_percentage < 50:
        return "High level of anomalies. Urgent investigation required."
    else:
        return "Critical level of anomalies. Immediate action necessary."
    

def calculate_dynamic_threshold(probabilities):
    return np.percentile(probabilities, DEFAULT_PERCENTILE)



def detect_anomalies_pcap(pcap_file, models, preprocessors, model_type, args):
    try:
        features_df = extract_features_from_pcap(pcap_file)
        features_df = engineer_features(features_df)
        
        if model_type == 'combine':
            rf_features = preprocess_for_inference(features_df.copy(), preprocessors['rf'], 'rf')
            xgb_features = preprocess_for_inference(features_df.copy(), preprocessors['xgboost'], 'xgboost')
            
            rf_probabilities = models['rf'].predict_proba(rf_features)[:, 1]
            xgb_probabilities = models['xgboost'].predict_proba(xgb_features)[:, 1]
            
            probabilities = combine_predictions(rf_probabilities, xgb_probabilities)
        else:
            features_df = preprocess_for_inference(features_df, preprocessors[model_type], model_type)
            probabilities = models[model_type].predict_proba(features_df)[:, 1]
        
        threshold = calculate_dynamic_threshold(probabilities)
        anomalies = features_df[probabilities > threshold]
        
        report = generate_report(anomalies, len(features_df), probabilities, args, features_df, model_type, threshold)
        print(report)

        return anomalies, probabilities
    except Exception as e:
        logging.error(f"An error occurred during anomaly detection: {str(e)}")
        logging.error("Please check your input data and ensure it matches the expected format.")
        logging.error(f"Current feature types:\n{features_df.dtypes}")
        return None, None
def detect_anomalies_live(interface, models, preprocessors, model_type, args):
    print(f"Monitoring live traffic on interface: {interface}")
    
    anomalies = []
    total_packets = 0
    error_count = 0

    iso_forest = IsolationForest(contamination=0.1, random_state=42)

    def packet_callback(packet):
        nonlocal total_packets, anomalies, error_count
        total_packets += 1
        
        try:
            if IP in packet:
                features = extract_features_from_packet(packet, packet.time)
                if features:
                    features_df = pd.DataFrame([features])
                    features_df = engineer_features(features_df)
                    
                    if model_type == 'combine':
                        rf_features = preprocess_for_inference(features_df.copy(), preprocessors['rf'], 'rf')
                        xgb_features = preprocess_for_inference(features_df.copy(), preprocessors['xgboost'], 'xgboost')
                        
                        rf_probability = models['rf'].predict_proba(rf_features)[0, 1]
                        xgb_probability = models['xgboost'].predict_proba(xgb_features)[0, 1]
                        
                        probability = combine_predictions(np.array([rf_probability]), np.array([xgb_probability]))[0]
                    else:
                        features_df = preprocess_for_inference(features_df, preprocessors[model_type], model_type)
                        probability = models[model_type].predict_proba(features_df)[0, 1]
                    
                    iso_forest_prediction = iso_forest.predict(features_df)
                    
                    anomaly_threshold = 0.7
                    is_anomaly = (probability > anomaly_threshold) or (iso_forest_prediction == -1)
                    
                    if is_anomaly:
                        print(f"Anomaly detected: {packet.summary()}")
                        print(f"Anomaly Probability: {probability:.4f}")
                        anomalies.append(features)

            if total_packets % 100 == 0:  # Generate report every 100 packets
                if model_type == 'combine':
                    probs = np.array([combine_predictions(
                        models['rf'].predict_proba(preprocess_for_inference(pd.DataFrame([a]), preprocessors['rf'], 'rf'))[:, 1],
                        models['xgboost'].predict_proba(preprocess_for_inference(pd.DataFrame([a]), preprocessors['xgboost'], 'xgboost'))[:, 1]
                    ) for a in anomalies])
                else:
                    probs = np.array([models[model_type].predict_proba(preprocess_for_inference(pd.DataFrame([a]), preprocessors[model_type], model_type))[:, 1] for a in anomalies])
                
                report = generate_report(anomalies, total_packets, probs, args, pd.DataFrame(anomalies), model_type)
                print(report)

        except Exception as e:
            error_count += 1
            print(f"Error processing packet: {e}")
            if error_count > 10:
                print("Too many errors. Stopping packet capture.")
                return True  # Stop sniffing

    try:
        sniff(iface=interface, prn=packet_callback, store=0, stop_filter=lambda _: error_count > 10)
    except KeyboardInterrupt:
        print("\nStopped packet capture.")
    finally:
        print(f"Total packets analyzed: {total_packets}")
        print(f"Total anomalies detected: {len(anomalies)}")
        print(f"Total errors encountered: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network Anomaly Detection System")
    parser.add_argument("--mode", choices=["detect_pcap", "detect_live", "list_interfaces", "train"], required=True)
    parser.add_argument("--model", choices=["rf", "xgboost", "combine"], help="Choose model: Random Forest (rf), XGBoost (xgboost), or Combined (combine). If not specified in train mode, both models will be trained.")
    parser.add_argument("--input", help="Input PCAP file for detection")
    parser.add_argument("--interface", help="Network interface for live detection")
    parser.add_argument("--detail", action="store_true", help="Generate detailed report")
    args = parser.parse_args()

    if args.mode == "list_interfaces":
        list_network_interfaces()
    elif args.mode == "train":
        train_and_evaluate_models(args)
    else:
        show_system_name(args.mode, args.input, args.interface)

        # Load the preprocessor and model
        if args.model in ['rf', 'combine', None]:
            rf_preprocessor = load_preprocessor('rf')
            rf_model = load_model('rf')
        if args.model in ['xgboost', 'combine', None]:
            xgb_preprocessor = load_preprocessor('xgboost')
            xgb_model = load_model('xgboost')

        models = {}
        preprocessors = {}
        if args.model in ['rf', 'combine', None]:
            models['rf'] = rf_model
            preprocessors['rf'] = rf_preprocessor
        if args.model in ['xgboost', 'combine', None]:
            models['xgboost'] = xgb_model
            preprocessors['xgboost'] = xgb_preprocessor

        if args.mode == "detect_pcap":
            if not args.input:
                print("Please provide an input PCAP file for detection.")
                exit(1)
            detect_anomalies_pcap(args.input, models, preprocessors, args.model, args)
        elif args.mode == "detect_live":
            if not args.interface:
                print("Please provide a network interface for live detection.")
                print("You can use the --mode list_interfaces option to see available interfaces.")
                exit(1)
            detect_anomalies_live(args.interface, models, preprocessors, args.model, args)