from scapy.all import IP, TCP, UDP
import pandas as pd
import numpy as np

# Constant for indicating missing values
MISSING_VALUE = -999

def extract_features_from_packet(packet):
    features = {}
    flags = {}  # To indicate whether a feature is actual or imputed

    if IP in packet:
        features['proto'] = packet[IP].proto
        features['sttl'] = packet[IP].ttl
        features['dttl'] = packet[IP].ttl  # Assuming same TTL for both directions
        features['sbytes'] = len(packet[IP])
        features['dbytes'] = 0  # We don't have info about return traffic in a single packet
    else:
        features['proto'] = MISSING_VALUE
        features['sttl'] = MISSING_VALUE
        features['dttl'] = MISSING_VALUE
        features['sbytes'] = MISSING_VALUE
        features['dbytes'] = MISSING_VALUE

    for key in ['proto', 'sttl', 'dttl', 'sbytes', 'dbytes']:
        flags[f'{key}_present'] = int(features[key] != MISSING_VALUE)

    if TCP in packet:
        features['sport'] = packet[TCP].sport
        features['dport'] = packet[TCP].dport
        features['swin'] = packet[TCP].window
        features['stcpb'] = packet[TCP].seq
        features['dtcpb'] = packet[TCP].ack
        features['state'] = 'CON'  # Simplified; you may need more logic to determine the actual state
    elif UDP in packet:
        features['sport'] = packet[UDP].sport
        features['dport'] = packet[UDP].dport
        features['swin'] = MISSING_VALUE
        features['stcpb'] = MISSING_VALUE
        features['dtcpb'] = MISSING_VALUE
        features['state'] = 'CON'
    else:
        features['sport'] = MISSING_VALUE
        features['dport'] = MISSING_VALUE
        features['swin'] = MISSING_VALUE
        features['stcpb'] = MISSING_VALUE
        features['dtcpb'] = MISSING_VALUE
        features['state'] = 'UNK'  # Unknown state

    for key in ['sport', 'dport', 'swin', 'stcpb', 'dtcpb']:
        flags[f'{key}_present'] = int(features[key] != MISSING_VALUE)

    # Initialize other features with MISSING_VALUE
    other_features = ['dur', 'sloss', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 
                      'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 
                      'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 
                      'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 
                      'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
                      'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 
                      'ct_dst_sport_ltm', 'ct_dst_src_ltm']

    for feature in other_features:
        features[feature] = MISSING_VALUE
        flags[f'{feature}_present'] = 0

    # Special cases
    if features['sbytes'] != MISSING_VALUE:
        features['smeansz'] = features['sbytes']
        flags['smeansz_present'] = 1
    
    features['Spkts'] = 1 if IP in packet else MISSING_VALUE
    flags['Spkts_present'] = int(IP in packet)

    features['attack_cat'] = 'Normal'  # Default to normal, model will predict if it's an attack
    flags['attack_cat_present'] = 1

    # Combine features and flags
    combined_features = {**features, **flags}
    
    return combined_features

def extract_features_from_pcap(pcap_file):
    from scapy.all import rdpcap
    
    packets = rdpcap(pcap_file)
    features_list = []
    
    for packet in packets:
        features = extract_features_from_packet(packet)
        if features:
            features_list.append(features)
    
    df = pd.DataFrame(features_list)
    
    # Replace MISSING_VALUE with NaN for better pandas handling
    df = df.replace(MISSING_VALUE, np.nan)
    
    return df

def preprocess_features(df):
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    # For numeric columns, impute with median
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # For categorical columns, impute with mode
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

    return df