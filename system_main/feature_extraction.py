from scapy.all import IP, TCP, UDP
import pandas as pd
import numpy as np
from collections import defaultdict

MISSING_VALUE = -999

connection_states = defaultdict(lambda: 'UNK')
connection_counts = defaultdict(int)
last_packet_time = {}

def extract_features_from_packet(packet, packet_time):
    features = {}
    flags = {}

    try:
        if IP in packet:
            features['proto'] = packet[IP].proto
            features['sttl'] = packet[IP].ttl
            features['dttl'] = packet[IP].ttl
            features['sbytes'] = len(packet[IP])
            features['dbytes'] = 0
            
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            
            conn_key = (src_ip, dst_ip)
            features['dur'] = packet_time - last_packet_time.get(conn_key, packet_time)
            last_packet_time[conn_key] = packet_time
        else:
            features['proto'] = features['sttl'] = features['dttl'] = features['sbytes'] = features['dbytes'] = features['dur'] = MISSING_VALUE
            src_ip = dst_ip = None

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
            features['swin'] = features['stcpb'] = features['dtcpb'] = MISSING_VALUE
            features['state'] = 'CON'
        else:
            features['sport'] = features['dport'] = features['swin'] = features['stcpb'] = features['dtcpb'] = MISSING_VALUE
            features['state'] = 'UNK'

        features['Sload'] = features['sbytes'] / features['dur'] if features['dur'] > 0 else 0
        features['Dload'] = features['dbytes'] / features['dur'] if features['dur'] > 0 else 0
        features['Spkts'] = 1
        features['Dpkts'] = 0
        features['smeansz'] = features['sbytes']
        features['dmeansz'] = 0

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

        for key in features:
            flags[f'{key}_present'] = int(features[key] != MISSING_VALUE)

    except Exception as e:
        print(f"Error extracting features from packet: {e}")
        return None

    return {**features, **flags}

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
    from scapy.all import rdpcap
    
    packets = rdpcap(pcap_file)
    features_list = []
    start_time = packets[0].time if packets else 0
    
    for packet in packets:
        features = extract_features_from_packet(packet, packet.time - start_time)
        if features:
            features_list.append(features)
    
    df = pd.DataFrame(features_list)
    df = df.replace(MISSING_VALUE, np.nan)
    
    return df

def preprocess_features(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
        df[col] = df[col].astype('category')

    return df