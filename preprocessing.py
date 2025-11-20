"""
Preprocessing utilities for the Network Intrusion Detection System
This module handles the same preprocessing pipeline used during training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_for_inference(input_data, scaler=None, return_scaler=False):
    """
    Preprocess input data for model inference.
    
    Args:
        input_data: Dictionary or DataFrame with input features
        scaler: Fitted StandardScaler (if None, will fit new one)
        return_scaler: Whether to return the scaler
    
    Returns:
        Preprocessed DataFrame ready for model prediction
    """
    # Convert dict to DataFrame if needed
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Drop IPs (not used in training)
    df = df.drop(['Source_IP', 'Destination_IP'], axis=1, errors='ignore')
    
    # Define categorical columns (as in training)
    categorical_cols = ['Request_Type', 'Protocol', 'User_Agent', 'Status', 'Port']
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Drop Intrusion (dropped during training for better generalization)
    df = df.drop(['Intrusion'], axis=1, errors='ignore')
    
    # Drop Scan_Type if present (it's the target)
    df = df.drop(['Scan_Type'], axis=1, errors='ignore')
    df = df.drop(['Scan_Type_Label'], axis=1, errors='ignore')
    
    # Expected columns after one-hot encoding (from training)
    expected_columns = [
        'Payload_Size',
        'Request_Type_DNS', 'Request_Type_FTP', 'Request_Type_HTTP', 
        'Request_Type_HTTPS', 'Request_Type_SMTP', 'Request_Type_SSH', 
        'Request_Type_Telnet',
        'Protocol_ICMP', 'Protocol_TCP', 'Protocol_UDP',
        'User_Agent_Mozilla/5.0', 'User_Agent_Nikto/2.1.6',
        'User_Agent_Wget/1.20.3', 'User_Agent_curl/7.68.0',
        'User_Agent_nmap/7.80', 'User_Agent_python-requests/2.25.1',
        'Status_Failure', 'Status_Success',
        'Port_21', 'Port_22', 'Port_23', 'Port_25', 'Port_53', 'Port_80',
        'Port_135', 'Port_443', 'Port_4444', 'Port_6667', 'Port_8080', 'Port_31337'
    ]
    
    # Add missing columns with zeros
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training order
    df = df.reindex(columns=expected_columns, fill_value=0)
    
    # Scale Payload_Size
    if 'Payload_Size' in df.columns:
        if scaler is None:
            scaler = StandardScaler()
            # Fit with dummy data (in production, load saved scaler)
            scaler.fit(df[['Payload_Size']])
        
        df['Payload_Size'] = scaler.transform(df[['Payload_Size']])
    
    if return_scaler:
        return df, scaler
    return df


def create_sample_input(request_type, protocol, status, port, payload_size, user_agent, intrusion=0):
    """
    Create a sample input dictionary from user inputs.
    
    Args:
        request_type: Request type (HTTP, HTTPS, etc.)
        protocol: Protocol (TCP, UDP, ICMP)
        status: Status (Success, Failure)
        port: Port number
        payload_size: Payload size in bytes
        user_agent: User agent string
        intrusion: Intrusion flag (0 or 1)
    
    Returns:
        Dictionary with all required fields
    """
    return {
        'Source_IP': '192.168.1.1',  # Dummy value, will be dropped
        'Destination_IP': '10.0.0.1',  # Dummy value, will be dropped
        'Request_Type': request_type,
        'Protocol': protocol,
        'Status': status,
        'Port': port,
        'Payload_Size': payload_size,
        'User_Agent': user_agent,
        'Intrusion': intrusion
    }

