"""
Batch Inference Module for Network Intrusion Detection

This module handles batch processing of network logs for classification.
"""

import pandas as pd
import numpy as np
from preprocessing import preprocess_for_inference
import joblib


def process_batch_csv(uploaded_file, model, scaler=None):
    """
    Process a CSV file with multiple network logs for batch classification.
    
    Parameters:
    -----------
    uploaded_file : Streamlit UploadedFile
        The uploaded CSV file
    model : trained model
        The classification model
    scaler : StandardScaler (optional)
        Pre-fitted scaler (if None, will fit new one)
    
    Returns:
    --------
    results_df : DataFrame with predictions and probabilities
    """
    
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Required columns
        required_cols = ['Request_Type', 'Protocol', 'Status', 'Port', 
                        'Payload_Size', 'User_Agent']
        
        # Check if all required columns are present
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Optional columns (will be added if missing)
        optional_cols = ['Source_IP', 'Destination_IP', 'Intrusion']
        for col in optional_cols:
            if col not in df.columns:
                df[col] = None if col in ['Source_IP', 'Destination_IP'] else 0
        
        # Process each row
        predictions = []
        probabilities = []
        
        for idx, row in df.iterrows():
            try:
                # Create input dictionary
                input_dict = {
                    'Source_IP': row.get('Source_IP', '192.168.1.1'),
                    'Destination_IP': row.get('Destination_IP', '10.0.0.1'),
                    'Request_Type': row['Request_Type'],
                    'Protocol': row['Protocol'],
                    'Status': row['Status'],
                    'Port': int(row['Port']),
                    'Payload_Size': float(row['Payload_Size']),
                    'User_Agent': row['User_Agent'],
                    'Intrusion': int(row.get('Intrusion', 0))
                }
                
                # Preprocess
                preprocessed = preprocess_for_inference(input_dict, scaler=scaler)
                
                # Predict
                pred = model.predict(preprocessed)[0]
                proba = model.predict_proba(preprocessed)[0]
                
                # Map prediction to class name
                label_mapping = {0: 'BotAttack', 1: 'Normal', 2: 'PortScan'}
                pred_class = label_mapping[pred]
                
                predictions.append(pred_class)
                probabilities.append({
                    'BotAttack_Prob': proba[0],
                    'Normal_Prob': proba[1],
                    'PortScan_Prob': proba[2],
                    'Confidence': max(proba)
                })
                
            except Exception as e:
                predictions.append(f"Error: {str(e)}")
                probabilities.append({
                    'BotAttack_Prob': None,
                    'Normal_Prob': None,
                    'PortScan_Prob': None,
                    'Confidence': None
                })
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['Predicted_Class'] = predictions
        results_df['BotAttack_Probability'] = [p['BotAttack_Prob'] for p in probabilities]
        results_df['Normal_Probability'] = [p['Normal_Prob'] for p in probabilities]
        results_df['PortScan_Probability'] = [p['PortScan_Prob'] for p in probabilities]
        results_df['Confidence'] = [p['Confidence'] for p in probabilities]
        
        return results_df, None
        
    except Exception as e:
        return None, str(e)


def generate_batch_summary(results_df):
    """
    Generate summary statistics for batch predictions.
    
    Parameters:
    -----------
    results_df : DataFrame with predictions
    
    Returns:
    --------
    summary_dict : Dictionary with summary statistics
    """
    
    summary = {
        'total_records': len(results_df),
        'predictions': {
            'Normal': len(results_df[results_df['Predicted_Class'] == 'Normal']),
            'BotAttack': len(results_df[results_df['Predicted_Class'] == 'BotAttack']),
            'PortScan': len(results_df[results_df['Predicted_Class'] == 'PortScan']),
            'Errors': len(results_df[results_df['Predicted_Class'].str.contains('Error', na=False)])
        },
        'confidence_stats': {
            'mean': results_df['Confidence'].mean() if 'Confidence' in results_df.columns else None,
            'min': results_df['Confidence'].min() if 'Confidence' in results_df.columns else None,
            'max': results_df['Confidence'].max() if 'Confidence' in results_df.columns else None,
            'std': results_df['Confidence'].std() if 'Confidence' in results_df.columns else None
        },
        'high_risk_count': len(results_df[
            (results_df['Predicted_Class'].isin(['BotAttack', 'PortScan'])) &
            (results_df['Confidence'] > 0.8)
        ]) if 'Confidence' in results_df.columns else 0
    }
    
    return summary


# Streamlit integration code (to be added to streamlit_app.py):
"""
# Add this to the Inference page in streamlit_app.py, replace the "Coming Soon" section:

st.markdown("---")
st.markdown("### 📝 Batch Inference")

uploaded_file = st.file_uploader(
    "Upload CSV file for batch classification",
    type=['csv'],
    help="CSV should contain: Request_Type, Protocol, Status, Port, Payload_Size, User_Agent"
)

if uploaded_file is not None:
    if st.session_state.model is not None:
        if st.button("🔍 Process Batch", use_container_width=True):
            with st.spinner("Processing batch file..."):
                from batch_inference import process_batch_csv, generate_batch_summary
                
                results_df, error = process_batch_csv(
                    uploaded_file, 
                    st.session_state.model
                )
                
                if error:
                    st.error(f"❌ Error processing file: {error}")
                else:
                    st.success(f"✅ Processed {len(results_df)} records successfully!")
                    
                    # Display summary
                    summary = generate_batch_summary(results_df)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", summary['total_records'])
                    with col2:
                        st.metric("Normal", summary['predictions']['Normal'])
                    with col3:
                        st.metric("BotAttack", summary['predictions']['BotAttack'], 
                                 delta=f"⚠️ {summary['predictions']['BotAttack']} threats")
                    with col4:
                        st.metric("PortScan", summary['predictions']['PortScan'],
                                 delta=f"⚠️ {summary['predictions']['PortScan']} threats")
                    
                    if summary['high_risk_count'] > 0:
                        st.warning(f"🚨 **{summary['high_risk_count']} high-confidence threats detected!**")
                    
                    # Display results table
                    st.markdown("### 📊 Batch Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    st.markdown("### 📈 Prediction Distribution")
                    pred_counts = results_df['Predicted_Class'].value_counts()
                    fig = px.pie(
                        values=pred_counts.values,
                        names=pred_counts.index,
                        title="Batch Prediction Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("⚠️ Model not loaded. Please ensure model file is available.")
"""

