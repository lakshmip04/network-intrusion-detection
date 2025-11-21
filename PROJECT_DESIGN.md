# 🛡️ Network Intrusion Detection System - Design Document

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [High-Level Design](#high-level-design)
3. [Low-Level Design](#low-level-design)
4. [System Architecture](#system-architecture)
5. [Data Flow](#data-flow)
6. [Model Architecture](#model-architecture)
7. [Technology Stack](#technology-stack)

---

## 🎯 Project Overview

### Problem Statement
Detect and classify network traffic into three categories:
- **Normal Traffic**: Legitimate network activity
- **BotAttack**: Automated malicious bot activity
- **PortScan**: Port scanning reconnaissance attacks

### Solution
A machine learning-based Intrusion Detection System (IDS) that:
- Analyzes network log features in real-time
- Classifies traffic using a Decision Tree classifier
- Provides interactive visualization and inference capabilities
- Supports both single and batch predictions

### Key Metrics
- **Accuracy**: 99.92%
- **ROC-AUC**: 99.97%
- **Dataset Size**: 8,846 records
- **Features**: 6 engineered features
- **Classes**: 3 (Normal, BotAttack, PortScan)

---

## 🏗️ High-Level Design

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Network Intrusion Detection System            │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Data Layer  │    │  ML Pipeline  │    │  UI Layer     │
│              │    │               │    │               │
│ • CSV Files  │───▶│ • Preprocess  │───▶│ • Streamlit   │
│ • Logs       │    │ • Train       │    │ • Dashboard   │
│ • Models     │    │ • Predict     │    │ • Inference   │
└───────────────┘    └───────────────┘    └───────────────┘
```

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Streamlit Web Application (Dashboard)                  │  │
│  │  • Data Overview    • EDA    • Model Results          │  │
│  │  • Inference        • About                           │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  streamlit_app.py                                       │  │
│  │  • Data Loading    • Model Loading                     │  │
│  │  • UI Components   • Visualization                     │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                        │
│  ┌──────────────────────┐  ┌────────────────────────────┐  │
│  │  preprocessing.py     │  │  batch_inference.py        │  │
│  │  • Feature Encoding  │  │  • Batch Processing        │  │
│  │  • Data Scaling      │  │  • CSV Handling            │  │
│  └──────────────────────┘  └────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                        MODEL LAYER                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Decision Tree Classifier (joblib)                     │  │
│  │  • Trained Model      • Feature Importances           │  │
│  │  • Prediction Logic  • Probability Scores            │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                         DATA LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Network_logs │  │ Time-Series  │  │ Model File  │      │
│  │    .csv      │  │   _logs.csv  │  │  .joblib    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### System Flow

```
┌─────────┐
│  User   │
└────┬────┘
     │
     │ 1. Access Dashboard
     ▼
┌─────────────────┐
│  Streamlit App  │
└────┬────────────┘
     │
     │ 2. Load Data & Model
     ▼
┌─────────────────┐
│  Data Pipeline  │──┐
└────┬────────────┘  │
     │               │
     │ 3. Preprocess │
     ▼               │
┌─────────────────┐  │
│  Feature        │  │
│  Engineering    │◄─┘
└────┬────────────┘
     │
     │ 4. Model Prediction
     ▼
┌─────────────────┐
│  Decision Tree  │
│  Classifier     │
└────┬────────────┘
     │
     │ 5. Results
     ▼
┌─────────────────┐
│  Visualization  │
│  & Alerts       │
└─────────────────┘
```

---

## 🔧 Low-Level Design

### 1. Data Preprocessing Pipeline

#### Input Data Structure
```python
{
    'Source_IP': str,           # e.g., '192.168.1.1'
    'Destination_IP': str,       # e.g., '10.0.0.1'
    'Port': int,                # e.g., 80
    'Request_Type': str,         # e.g., 'HTTP', 'HTTPS', 'FTP'
    'Protocol': str,             # e.g., 'TCP', 'UDP', 'ICMP'
    'Payload_Size': int,         # e.g., 1500 (bytes)
    'User_Agent': str,           # e.g., 'Mozilla/5.0'
    'Status': str,               # 'Success' or 'Failure'
    'Intrusion': int,            # 0 or 1 (dropped during training)
    'Scan_Type': str             # Target: 'Normal', 'BotAttack', 'PortScan'
}
```

#### Preprocessing Steps

```
Raw Input Data
    │
    ├─▶ Drop IPs (Source_IP, Destination_IP)
    │   Reason: High cardinality, not generalizable
    │
    ├─▶ Drop Intrusion Feature
    │   Reason: Prevent overfitting, encourage generalization
    │
    ├─▶ Categorical Encoding
    │   ├─ Request_Type → Category Code (0-6)
    │   ├─ Protocol → Category Code (0-2)
    │   ├─ User_Agent → Category Code (0-5)
    │   ├─ Status → Category Code (0-1)
    │   └─ Port → Category Code (0-11)
    │
    ├─▶ Numerical Scaling
    │   └─ Payload_Size → StandardScaler
    │      Formula: (x - mean) / std
    │      Mean: 1598.76, Std: 915.62
    │
    └─▶ Feature Selection
        Final Features: ['Port', 'Request_Type', 'Protocol', 
                        'Payload_Size', 'User_Agent', 'Status']
```

#### Category Mappings

```python
Request_Type: ['DNS', 'FTP', 'HTTP', 'HTTPS', 'SMTP', 'SSH', 'Telnet']
              [  0,     1,     2,       3,       4,      5,      6   ]

Protocol:     ['ICMP', 'TCP', 'UDP']
              [  0,      1,     2  ]

User_Agent:   ['Mozilla/5.0', 'Nikto/2.1.6', 'Wget/1.20.3', 
               'curl/7.68.0', 'nmap/7.80', 'python-requests/2.25.1']
              [     0,           1,             2,
                     3,           4,             5            ]

Status:       ['Failure', 'Success']
              [   0,         1     ]

Port:         [21, 22, 23, 25, 53, 80, 135, 443, 4444, 6667, 8080, 31337]
              [ 0,  1,  2,  3,  4,  5,   6,   7,    8,    9,   10,    11]
```

### 2. Model Architecture

#### Decision Tree Classifier

```
                    Decision Tree Root
                    (Feature: Payload_Size)
                           │
            ┌───────────────┴───────────────┐
            │                               │
      Payload_Size < threshold      Payload_Size >= threshold
            │                               │
      [Subtree A]                      [Subtree B]
            │                               │
    (Feature: Status)              (Feature: Protocol)
            │                               │
      ┌─────┴─────┐                   ┌─────┴─────┐
      │           │                   │           │
  Status=0   Status=1            Protocol=0  Protocol=1
  (Failure)  (Success)           (ICMP)     (TCP/UDP)
      │           │                   │           │
  [Leaf]      [Leaf]              [Leaf]      [Leaf]
  Class:      Class:              Class:      Class:
  PortScan    Normal              BotAttack   Normal
```

#### Model Parameters

```python
DecisionTreeClassifier(
    criterion='gini',           # Splitting criterion
    max_depth=None,             # Unlimited depth
    min_samples_split=2,        # Minimum samples to split
    min_samples_leaf=1,         # Minimum samples in leaf
    random_state=42             # Reproducibility
)
```

#### Model Characteristics
- **Type**: Supervised Learning - Classification
- **Algorithm**: Decision Tree (CART - Classification and Regression Tree)
- **Splitting Criterion**: Gini Impurity
- **Number of Features**: 6
- **Number of Classes**: 3
- **Output**: Probability distribution over classes

#### Feature Importance (Expected Order)
1. **Payload_Size** (Highest) - Strong discriminator
2. **Status** - Success/Failure patterns
3. **Protocol** - TCP/UDP/ICMP usage
4. **Request_Type** - HTTP/HTTPS/FTP patterns
5. **Port** - Port number patterns
6. **User_Agent** (Lowest) - Client identification

### 3. Training Pipeline

```
Training Dataset (8,846 records)
    │
    ├─▶ Train-Test Split (70-30)
    │   ├─ Training: 6,192 records
    │   └─ Testing: 2,654 records
    │   └─ Stratified: Maintains class distribution
    │
    ├─▶ Preprocessing
    │   └─ (Same as inference pipeline)
    │
    ├─▶ Model Training
    │   └─ DecisionTreeClassifier.fit(X_train, y_train)
    │
    ├─▶ Model Evaluation
    │   ├─ Accuracy: 99.92%
    │   ├─ Precision (weighted): 99.93%
    │   ├─ Recall (weighted): 99.92%
    │   ├─ F1-Score (weighted): 99.92%
    │   └─ ROC-AUC: 99.97%
    │
    └─▶ Model Persistence
        └─ joblib.dump(model, 'network_logs_decision_tree_model.joblib')
```

### 4. Inference Pipeline

```
User Input (Streamlit Form)
    │
    ├─▶ create_sample_input()
    │   └─ Creates input dictionary
    │
    ├─▶ preprocess_for_inference()
    │   ├─ Convert to DataFrame
    │   ├─ Drop unnecessary columns
    │   ├─ Encode categoricals
    │   ├─ Scale Payload_Size
    │   └─ Reorder features
    │
    ├─▶ model.predict()
    │   └─ Returns class index (0, 1, or 2)
    │
    ├─▶ model.predict_proba()
    │   └─ Returns probability distribution
    │
    └─▶ Display Results
        ├─ Predicted class
        ├─ Probability scores
        ├─ Visualizations
        └─ Security alerts
```

### 5. Class Mapping

```python
Label Encoding:
    0 → BotAttack    (Automated malicious activity)
    1 → Normal       (Legitimate traffic)
    2 → PortScan     (Reconnaissance activity)
```

### 6. Decision Logic

```
IF Payload_Size < threshold_1:
    IF Status == Failure:
        → PortScan (High confidence)
    ELSE:
        → Normal (Medium confidence)
ELSE:
    IF Protocol == ICMP:
        → BotAttack (High confidence)
    ELSE:
        IF Request_Type == HTTP/HTTPS:
            → Normal (High confidence)
        ELSE:
            → BotAttack (Medium confidence)
```

---

## 🏛️ System Architecture

### Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Application                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Pages      │  │  Components  │  │  Utilities   │    │
│  │              │  │              │  │              │    │
│  │ • Overview   │  │ • Forms      │  │ • load_data()│    │
│  │ • EDA        │  │ • Charts     │  │ • load_model()│    │
│  │ • Results    │  │ • Tables     │  │ • Visualize  │    │
│  │ • Inference  │  │ • Alerts     │  │              │    │
│  │ • About      │  │              │  │              │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │             │
└─────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Processing Module                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  preprocessing.py                                    │  │
│  │  • encode_categorical_value()                        │  │
│  │  • preprocess_for_inference()                        │  │
│  │  • create_sample_input()                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  batch_inference.py                                   │  │
│  │  • process_batch_csv()                               │  │
│  │  • generate_batch_summary()                          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                               │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Decision Tree Classifier                            │  │
│  │  • predict()      → Class prediction                │  │
│  │  • predict_proba() → Probability scores              │  │
│  │  • feature_importances_ → Feature importance         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌─────────────┐
│ Network Log │
│   (CSV)     │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Data Loading    │
│  (pandas)        │
└──────┬───────────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  Training    │  │  Inference   │
│  Pipeline    │  │  Pipeline    │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  Preprocess  │  │  Preprocess  │
│  & Train     │  │  & Predict   │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  Save Model  │  │  Display     │
│  (.joblib)   │  │  Results     │
└──────────────┘  └──────────────┘
```

---

## 📊 Data Flow

### Training Phase

```
1. Data Ingestion
   Network_logs.csv (8,846 records)
   │
   ├─▶ Load with pandas
   ├─▶ Check data types
   └─▶ Validate completeness

2. Data Preprocessing
   │
   ├─▶ Drop Source_IP, Destination_IP
   ├─▶ Convert categoricals to category type
   ├─▶ Encode with .cat.codes
   ├─▶ Scale Payload_Size
   └─▶ Drop Intrusion feature

3. Feature Engineering
   │
   └─▶ Final 6 features:
       • Port (category code)
       • Request_Type (category code)
       • Protocol (category code)
       • Payload_Size (scaled)
       • User_Agent (category code)
       • Status (category code)

4. Model Training
   │
   ├─▶ Split: 70% train, 30% test
   ├─▶ Stratified split (maintains distribution)
   ├─▶ Train DecisionTreeClassifier
   └─▶ Evaluate on test set

5. Model Persistence
   │
   └─▶ Save as .joblib file
```

### Inference Phase

```
1. User Input
   Streamlit form fields
   │
   ├─▶ Request_Type (dropdown)
   ├─▶ Protocol (dropdown)
   ├─▶ Status (dropdown)
   ├─▶ Port (number input)
   ├─▶ Payload_Size (number input)
   └─▶ User_Agent (dropdown)

2. Input Processing
   │
   ├─▶ create_sample_input() → Dictionary
   └─▶ preprocess_for_inference() → DataFrame

3. Feature Encoding
   │
   ├─▶ Map categoricals to codes
   ├─▶ Scale Payload_Size
   └─▶ Reorder to match training

4. Prediction
   │
   ├─▶ model.predict() → Class (0, 1, or 2)
   └─▶ model.predict_proba() → Probabilities

5. Result Display
   │
   ├─▶ Map class to name
   ├─▶ Show probabilities
   ├─▶ Display visualizations
   └─▶ Show security alerts
```

---

## 🛠️ Technology Stack

### Frontend
- **Streamlit 1.51.0**: Web application framework
- **Plotly 6.3.1**: Interactive visualizations
- **Matplotlib 3.10.7**: Static plots
- **Seaborn 0.13.2**: Statistical visualizations

### Backend
- **Python 3.13**: Programming language
- **Pandas 2.3.3**: Data manipulation
- **NumPy 2.2.6**: Numerical computing

### Machine Learning
- **Scikit-learn 1.7.0**: ML algorithms and utilities
  - DecisionTreeClassifier
  - StandardScaler
  - train_test_split
  - Metrics (accuracy, precision, recall, F1, ROC-AUC)
- **Joblib 1.5.1**: Model persistence

### Data Processing
- **CSV Files**: Data storage
- **Category Encoding**: Pandas categorical encoding
- **Feature Scaling**: StandardScaler

---

## 📈 Performance Metrics

### Model Performance
- **Accuracy**: 99.92%
- **Precision (Weighted)**: 99.93%
- **Recall (Weighted)**: 99.92%
- **F1-Score (Weighted)**: 99.92%
- **ROC-AUC (Multi-class)**: 99.97%

### Per-Class Performance
```
Class        Precision  Recall  F1-Score
----------------------------------------
BotAttack    1.00       0.99    0.99
Normal       1.00       1.00    1.00
PortScan     1.00       1.00    1.00
```

### Dataset Statistics
- **Total Records**: 8,846
- **Training Set**: 6,192 (70%)
- **Test Set**: 2,654 (30%)
- **Features**: 6
- **Classes**: 3

### Class Distribution
- **Normal**: 90.4% (7,998 records)
- **BotAttack**: 5.4% (478 records)
- **PortScan**: 4.2% (370 records)

---

## 🔐 Security Considerations

### Model Security
- Model file integrity checks
- Input validation and sanitization
- Error handling for malformed inputs

### Data Privacy
- IP addresses dropped (not used in training)
- No PII (Personally Identifiable Information) stored
- Anonymized network logs

### Deployment Security
- Input validation
- Error handling
- Secure model loading

---

## 🚀 Deployment Architecture

### Current Deployment
```
Local Machine
    │
    ├─▶ Python Environment
    ├─▶ Streamlit Server (localhost:8501)
    ├─▶ Model File (.joblib)
    └─▶ Data Files (.csv)
```

### Production-Ready Deployment (Future)
```
Cloud Server / Container
    │
    ├─▶ Docker Container
    ├─▶ Streamlit App
    ├─▶ Model Registry
    ├─▶ Database (PostgreSQL)
    └─▶ API Gateway (FastAPI)
```

---

## 📝 Key Design Decisions

### 1. Why Decision Tree?
- **Interpretability**: Easy to understand and explain
- **Performance**: 99.92% accuracy
- **Speed**: Fast inference time
- **No assumptions**: Non-parametric model

### 2. Why Category Codes over One-Hot Encoding?
- **Dimensionality**: 6 features vs 30+ with one-hot
- **Efficiency**: Faster training and inference
- **Memory**: Lower memory footprint
- **Model simplicity**: Simpler decision boundaries

### 3. Why Drop IP Addresses?
- **Generalization**: IPs are too specific
- **Overfitting prevention**: Avoid memorizing IPs
- **Scalability**: Works with any network

### 4. Why Drop Intrusion Feature?
- **Prevent leakage**: Intrusion is too correlated with target
- **Better generalization**: Model learns from other features
- **Real-world applicability**: Intrusion flag may not be available

---

## 🔄 Future Enhancements

1. **Real-time Processing**: Stream processing for live network logs
2. **Model Retraining**: Automated retraining pipeline
3. **Ensemble Methods**: Combine multiple models
4. **Deep Learning**: Neural networks for complex patterns
5. **API Deployment**: RESTful API for integration
6. **Database Integration**: Store predictions and logs
7. **Alert System**: Automated notifications for threats
8. **Model Monitoring**: Track model performance over time

---

## 📚 References

- Scikit-learn Documentation: Decision Tree Classifier
- Streamlit Documentation: Web app framework
- Network Security Best Practices
- Machine Learning Model Deployment Patterns

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Author**: Network Intrusion Detection System Project


