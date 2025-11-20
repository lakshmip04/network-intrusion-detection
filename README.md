# Network Intrusion Detection System - Streamlit Dashboard

A comprehensive Streamlit web application for visualizing and analyzing network intrusion detection results, including data exploration, model performance analysis, and real-time inference.

## 🚀 Features

### 📊 Data Overview
- Dataset statistics and information
- Feature descriptions
- Data preview and data type analysis
- Target variable distribution

### 📈 Exploratory Data Analysis (EDA)
- Feature distributions (numerical and categorical)
- Correlation analysis and heatmaps
- Feature relationships by scan type
- Target variable insights and class imbalance analysis

### 🤖 Model Results
- Performance comparison of all trained models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree ⭐ (Selected)
  - Random Forest
  - XGBoost
  - Multi-Layer Perceptron (MLP)
- Detailed metrics for each model
- Feature importance visualization

### 🔮 Inference
- Interactive form for single prediction
- Real-time classification with probability scores
- Preprocessing pipeline integration
- Security alerts for detected threats

## 📦 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the following files in the same directory:
   - `Network_logs.csv` - The dataset
   - `network_logs_decision_tree_model.joblib` - The trained model (optional, for inference)
   - `streamlit_app.py` - The main application
   - `preprocessing.py` - Preprocessing utilities

## 🏃 Running the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
.
├── streamlit_app.py          # Main Streamlit application
├── preprocessing.py          # Preprocessing utilities for inference
├── requirements.txt          # Python dependencies
├── Network_logs.csv          # Dataset
├── network_logs_decision_tree_model.joblib  # Trained model (optional)
└── README.md                 # This file
```

## 🎯 Dataset Information

The dataset contains network log entries with the following features:

- **Source_IP / Destination_IP**: IP addresses (dropped during training)
- **Port**: Network port number
- **Request_Type**: Type of request (HTTP, HTTPS, FTP, SSH, SMTP, DNS, Telnet)
- **Protocol**: Network protocol (TCP, UDP, ICMP)
- **Payload_Size**: Size of data payload in bytes
- **User_Agent**: Client user agent string
- **Status**: Request status (Success/Failure)
- **Intrusion**: Binary flag (0 = Normal, 1 = Attack)
- **Scan_Type**: Target variable (Normal, BotAttack, PortScan)

## 🔍 Model Performance

All models achieved excellent performance:
- **Best Model**: Decision Tree Classifier
- **Accuracy**: 99.92%
- **ROC-AUC**: 99.97%
- All models: >97% accuracy

## 📝 Notes

- The Intrusion feature was dropped during training to encourage better generalization
- Class imbalance exists: Normal (90.4%), BotAttack (5.4%), PortScan (4.2%)
- One-hot encoding was applied to categorical features
- Payload size was standardized using StandardScaler

## 🛠️ Technology Stack

- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Plotly, Matplotlib, Seaborn

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements.

