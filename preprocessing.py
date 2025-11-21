"""
Preprocessing utilities for the Network Intrusion Detection System
This module handles the same preprocessing pipeline used during training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib


# Define the category mappings based on training data
# These should match the order from the training dataset
CATEGORY_MAPPINGS = {
    'Request_Type': ['DNS', 'FTP', 'HTTP', 'HTTPS', 'SMTP', 'SSH', 'Telnet'],
    'Protocol': ['ICMP', 'TCP', 'UDP'],
    'User_Agent': ['Mozilla/5.0', 'Nikto/2.1.6', 'Wget/1.20.3', 'curl/7.68.0', 
                   'nmap/7.80', 'python-requests/2.25.1'],
    'Status': ['Failure', 'Success'],
    'Port': [21, 22, 23, 25, 53, 80, 135, 443, 4444, 6667, 8080, 31337]
}


def encode_categorical_value(value, category_list):
    """
    Encode a categorical value to its category code.
    
    Args:
        value: The categorical value to encode
        category_list: List of all possible categories in order
    
    Returns:
        The category code (integer)
    """
    try:
        return category_list.index(value)
    except ValueError:
        # If value not in list, return -1 or use a default
        # For inference, we'll use the first category as default
        return 0


def preprocess_for_inference(input_data, scaler=None, return_scaler=False):
    """
    Preprocess input data for model inference.
    This matches the preprocessing used during training: category codes, not one-hot encoding.
    
    Args:
        input_data: Dictionary or DataFrame with input features
        scaler: Fitted StandardScaler (if None, will fit new one - not recommended)
        return_scaler: Whether to return the scaler
    
    Returns:
        Preprocessed DataFrame ready for model prediction with columns:
        ['Port', 'Request_Type', 'Protocol', 'Payload_Size', 'User_Agent', 'Status']
    """
    # Convert dict to DataFrame if needed
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Drop IPs (not used in training)
    df = df.drop(['Source_IP', 'Destination_IP'], axis=1, errors='ignore')
    
    # Drop Intrusion (dropped during training for better generalization)
    df = df.drop(['Intrusion'], axis=1, errors='ignore')
    
    # Drop Scan_Type if present (it's the target)
    df = df.drop(['Scan_Type'], axis=1, errors='ignore')
    df = df.drop(['Scan_Type_Label'], axis=1, errors='ignore')
    
    # Encode categorical columns using category codes (matching training)
    categorical_cols = ['Request_Type', 'Protocol', 'User_Agent', 'Status', 'Port']
    
    for col in categorical_cols:
        if col in df.columns:
            if col == 'Port':
                # Port is numeric, but we need to map it to category code
                df[col] = df[col].astype(int)
                df[col] = df[col].apply(lambda x: encode_categorical_value(x, CATEGORY_MAPPINGS[col]))
            else:
                # For string categories, encode to category code
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: encode_categorical_value(x, CATEGORY_MAPPINGS[col]))
    
    # Ensure Payload_Size is numeric
    if 'Payload_Size' in df.columns:
        df['Payload_Size'] = pd.to_numeric(df['Payload_Size'], errors='coerce').fillna(0)
    
    # Scale Payload_Size (using the same scaler from training if available)
    # Note: In production, you should load the saved scaler
    if 'Payload_Size' in df.columns:
        if scaler is None:
            # Use statistics from training data (calculated from Network_logs.csv)
            # Mean: 1598.76, Std: 915.62
            mean_payload = 1598.76
            std_payload = 915.62
            df['Payload_Size'] = (df['Payload_Size'] - mean_payload) / std_payload
        else:
            df['Payload_Size'] = scaler.transform(df[['Payload_Size']])
    
    # Select and order columns to match training
    expected_columns = ['Port', 'Request_Type', 'Protocol', 'Payload_Size', 'User_Agent', 'Status']
    
    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training order exactly
    df = df[expected_columns]
    
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
