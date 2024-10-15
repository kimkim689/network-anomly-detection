from scapy.all import IP, TCP, UDP, rdpcap
import pandas as pd
import numpy as np
from collections import defaultdict
from decimal import Decimal

# Constant for indicating missing values
MISSING_VALUE = -999

# Global variables for connection tracking
connection_states = defaultdict(lambda: 'UNK')
connection_counts = defaultdict(int)
last_packet_time = {}

def extract_features_from_packet(packet, packet_time):
    features = {}
    flags = {}  # To indicate whether a feature is actual or imputed

    try:
        if IP in packet:
            features['proto'] = packet[IP].proto
            features['sttl'] = packet[IP].ttl
            features['dttl'] = packet[IP].ttl  # Assuming same TTL for both directions
            features['sbytes'] = len(packet[IP])
            features['dbytes'] = 0  # We don't have info about return traffic in a single packet
            
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            
            # Calculate duration
            conn_key = (src_ip, dst_ip)
            if conn_key in last_packet_time:
                features['dur'] = packet_time - last_packet_time[conn_key]
            else:
                features['dur'] = 0
            last_packet_time[conn_key] = packet_time
        else:
            features['proto'] = MISSING_VALUE
            features['sttl'] = MISSING_VALUE
            features['dttl'] = MISSING_VALUE
            features['sbytes'] = MISSING_VALUE
            features['dbytes'] = MISSING_VALUE
            features['dur'] = MISSING_VALUE
            src_ip = dst_ip = None

        for key in ['proto', 'sttl', 'dttl', 'sbytes', 'dbytes', 'dur']:
            flags[f'{key}_present'] = int(features[key] != MISSING_VALUE)

        if TCP in packet:
            features['sport'] = packet[TCP].sport
            features['dport'] = packet[TCP].dport
            features['swin'] = packet[TCP].window
            features['stcpb'] = packet[TCP].seq
            features['dtcpb'] = packet[TCP].ack
            features['state'] = update_tcp_state(packet[TCP].flags)
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
            features['state'] = 'UNK'

        for key in ['sport', 'dport', 'swin', 'stcpb', 'dtcpb']:
            flags[f'{key}_present'] = int(features[key] != MISSING_VALUE)
        flags['state_present'] = 1  # Always consider state as present

        # Calculate additional features
        features['Sload'] = features['sbytes'] / features['dur'] if features['dur'] > 0 else 0
        features['Dload'] = features['dbytes'] / features['dur'] if features['dur'] > 0 else 0
        features['Spkts'] = 1
        features['Dpkts'] = 0
        features['smeansz'] = features['sbytes']
        features['dmeansz'] = 0

        # Connection-based features
        if src_ip and dst_ip:
            conn_key = (src_ip, dst_ip, features['sport'], features['dport'])
            features['ct_state_ttl'] = update_connection_count(conn_key)
            features['ct_srv_src'] = update_connection_count((src_ip, features['sport']))
            features['ct_srv_dst'] = update_connection_count((dst_ip, features['dport']))
            features['ct_dst_ltm'] = update_connection_count((dst_ip,))
            features['ct_src_ltm'] = update_connection_count((src_ip,))
            features['ct_src_dport_ltm'] = update_connection_count((src_ip, features['dport']))
            features['ct_dst_sport_ltm'] = update_connection_count((dst_ip, features['sport']))
            features['ct_dst_src_ltm'] = update_connection_count((dst_ip, src_ip))
        else:
            for feature in ['ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 
                            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']:
                features[feature] = MISSING_VALUE

        # Set flags for new features
        for feature in ['Sload', 'Dload', 'Spkts', 'Dpkts', 'smeansz', 'dmeansz', 
                        'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 
                        'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']:
            flags[f'{feature}_present'] = int(features[feature] != MISSING_VALUE)

        # Combine features and flags
        combined_features = {**features, **flags}

    except Exception as e:
        print(f"Error extracting features from packet: {e}")
        return None

    return combined_features
def update_tcp_state(flags):
    if flags.S:
        return 'SYN'
    elif flags.SA:
        return 'SYN-ACK'
    elif flags.A:
        return 'ACK'
    elif flags.F:
        return 'FIN'
    elif flags.R:
        return 'RST'
    else:
        return 'CON'

def update_connection_count(key):
    connection_counts[key] += 1
    return connection_counts[key]

def extract_features_from_pcap(pcap_file):
    try:
        packets = rdpcap(pcap_file)
        features_list = []
        start_time = packets[0].time if packets else 0
        
        for packet in packets:
            features = extract_features_from_packet(packet, packet.time - start_time)
            if features:
                features_list.append(features)
        
        if not features_list:
            print("No features could be extracted from the PCAP file.")
            return None
        
        df = pd.DataFrame(features_list)
        
        # Calculate statistical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[f'{col}_mean'] = df[col].expanding().mean()
            df[f'{col}_std'] = df[col].expanding().std()
        
        return df
    except Exception as e:
        print(f"Error reading PCAP file: {e}")
        return None
def preprocess_live_data(df, preprocessor, selected_features):
    # Only use features that are both in the DataFrame and in selected_features
    available_features = list(set(df.columns) & set(selected_features))
    df = df[available_features]
    
    # Encode categorical columns
    for col, encoder in preprocessor['label_encoders'].items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ else -1)

    # We'll skip imputation to avoid introducing potentially misleading data
    
    return df, available_features


def preprocess_features(df):
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    # For numeric columns, impute with median
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # For categorical columns, impute with mode and convert to string
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
        df[col] = df[col].astype(str)  # Convert to string to ensure hashability

    # Convert Decimal columns to float
    for col in df.columns:
        if df[col].dtype == object:  # Check if column contains objects
            if df[col].apply(lambda x: isinstance(x, Decimal)).any():  # Check if any value is Decimal
                df[col] = df[col].astype(float)

    return df