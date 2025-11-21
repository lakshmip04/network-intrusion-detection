import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    """Load the network logs dataset"""
    # Try multiple possible paths
    data_paths = [
        "Network_logs.csv",  # Current directory
        os.path.join(os.getcwd(), "Network_logs.csv"),  # Explicit current directory
    ]
    
    # Try to get script directory if __file__ is available
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_paths.insert(0, os.path.join(script_dir, "Network_logs.csv"))
    except:
        pass
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                continue
    
    st.error("Network_logs.csv file not found. Please ensure the file is in the same directory.")
    return None

# Load model function with caching
@st.cache_resource
def load_model():
    """Load the saved Decision Tree model"""
    # Try multiple possible paths
    model_paths = [
        "network_logs_decision_tree_model.joblib",  # Current directory
        os.path.join(os.getcwd(), "network_logs_decision_tree_model.joblib"),  # Explicit current directory
    ]
    
    # Try to get script directory if __file__ is available
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_paths.insert(0, os.path.join(script_dir, "network_logs_decision_tree_model.joblib"))
    except:
        pass
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model
            except Exception as e:
                continue
    
    st.warning("⚠️ Model file not found. Inference will be disabled.")
    st.info(f"💡 Looking for model in: {os.getcwd()}")
    return None

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = load_data()

if 'model' not in st.session_state:
    st.session_state.model = load_model()

# Sidebar navigation
st.sidebar.title("🛡️ Network IDS Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["📊 Data Overview", "📈 Exploratory Data Analysis", "🤖 Model Results", "🔮 Inference", "📅 Time Series Forecast", "📚 About"]
)

# Main content based on page selection
if page == "📊 Data Overview":
    st.markdown('<div class="main-header">Network Intrusion Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Dataset Overview
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.markdown("---")
        
        # Dataset Description
        st.header("📖 Dataset Description")
        
        st.markdown("""
        This project focuses on detecting different types of network activity—specifically distinguishing 
        between **Normal traffic**, **Bot Attacks**, and **Port Scan Attacks**—based on structured network logs.
        
        **Objective:** Using the `Scan_Type` column as our target label, we aim to build a multiclass classification 
        model that can identify the nature of each network request.
        """)
        
        st.markdown("### Dataset Features:")
        feature_desc = {
            "Source_IP": "Source IP address of the network request",
            "Destination_IP": "Destination IP address",
            "Port": "Network port number used for the connection",
            "Request_Type": "Type of request (HTTP, HTTPS, FTP, SSH, etc.)",
            "Protocol": "Network protocol used (TCP, UDP, ICMP)",
            "Payload_Size": "Size of the data payload in bytes",
            "User_Agent": "User agent string identifying the client",
            "Status": "Request status (Success/Failure)",
            "Intrusion": "Binary target (0 = Normal, 1 = Attack)",
            "Scan_Type": "Traffic type classification (Normal, PortScan, BotAttack)"
        }
        
        feature_df = pd.DataFrame(list(feature_desc.items()), columns=["Feature", "Description"])
        st.dataframe(feature_df, use_container_width=True)
        
        st.markdown("---")
        
        # Data Preview
        st.header("👀 Data Preview")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            num_rows = st.slider("Number of rows to display", 5, 100, 10)
        with col2:
            st.empty()
        
        st.dataframe(df.head(num_rows), use_container_width=True)
        
        st.markdown("---")
        
        # Data Info
        st.header("ℹ️ Data Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                "Column": df.dtypes.index,
                "Data Type": df.dtypes.values.astype(str)
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Missing Values")
            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing Count": df.isnull().sum().values,
                "Missing %": (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df["Missing Count"] > 0]
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values in the dataset!")
        
        st.markdown("---")
        
        # Target Distribution
        st.header("🎯 Target Variable Distribution")
        
        if 'Scan_Type' in df.columns:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    df['Scan_Type'].value_counts().reset_index(),
                    x='Scan_Type',
                    y='count',
                    color='Scan_Type',
                    title="Scan Type Distribution",
                    labels={'Scan_Type': 'Scan Type', 'count': 'Count'},
                    color_discrete_map={
                        'Normal': '#2ecc71',
                        'BotAttack': '#e74c3c',
                        'PortScan': '#f39c12'
                    }
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Distribution Stats")
                scan_type_counts = df['Scan_Type'].value_counts()
                scan_type_pct = df['Scan_Type'].value_counts(normalize=True) * 100
                
                for scan_type, count in scan_type_counts.items():
                    pct = scan_type_pct[scan_type]
                    st.metric(
                        label=scan_type,
                        value=f"{count:,} ({pct:.1f}%)"
                    )
                
                st.info("⚠️ Dataset shows class imbalance. Normal traffic dominates (90.4%), while BotAttack (5.4%) and PortScan (4.2%) are underrepresented.")
        
        # Statistical Summary
        st.markdown("---")
        st.header("📊 Statistical Summary")
        
        st.subheader("Numerical Features")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Categorical Features")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:5]:  # Show first 5 categorical columns
            with st.expander(f"Value counts for {col}"):
                value_counts = df[col].value_counts().head(20)
                st.dataframe(value_counts)

elif page == "📈 Exploratory Data Analysis":
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # EDA Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Distributions", 
            "🔗 Relationships", 
            "📈 Feature Analysis", 
            "🎯 Target Insights"
        ])
        
        with tab1:
            st.header("Feature Distributions")
            
            # Numerical features
            st.subheader("Numerical Features Distribution")
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if 'Intrusion' in numerical_cols:
                numerical_cols.remove('Intrusion')
            if 'Scan_Type' in df.columns:
                pass
            
            selected_num = st.selectbox("Select numerical feature", numerical_cols)
            
            if selected_num:
                fig = px.histogram(
                    df,
                    x=selected_num,
                    nbins=50,
                    title=f"Distribution of {selected_num}",
                    labels={selected_num: selected_num.replace('_', ' ')}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{df[selected_num].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[selected_num].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{df[selected_num].std():.2f}")
            
            # Categorical features
            st.subheader("Categorical Features Distribution")
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            selected_cat = st.selectbox("Select categorical feature", categorical_cols)
            
            if selected_cat:
                value_counts = df[selected_cat].value_counts().head(15)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution of {selected_cat}",
                    labels={'x': selected_cat.replace('_', ' '), 'y': 'Count'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Feature Relationships")
            
            # Correlation heatmap
            if 'Scan_Type' in df.columns:
                st.subheader("Correlation Analysis")
                
                # Encode categoricals for correlation
                df_encoded = df.copy()
                for col in ['Request_Type', 'Protocol', 'User_Agent', 'Status', 'Scan_Type']:
                    if col in df_encoded.columns:
                        df_encoded[col] = df_encoded[col].astype('category').cat.codes
                
                df_encoded = df_encoded.drop(['Source_IP', 'Destination_IP'], axis=1, errors='ignore')
                
                corr_matrix = df_encoded.select_dtypes(include=[np.number]).corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Heatmap",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Features with strong correlation (>0.7 or <-0.7) may indicate multicollinearity.")
            
            # Request Type by Scan Type
            st.subheader("Request Type by Scan Type")
            if 'Request_Type' in df.columns and 'Scan_Type' in df.columns:
                cross_tab = pd.crosstab(df['Request_Type'], df['Scan_Type'])
                fig = px.bar(
                    cross_tab,
                    barmode='group',
                    title="Request Type Distribution by Scan Type",
                    labels={'value': 'Count', 'Request_Type': 'Request Type'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Feature Analysis by Scan Type")
            
            if 'Scan_Type' in df.columns:
                # Payload Size Analysis
                st.subheader("Payload Size by Scan Type")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Box plot
                    fig = px.box(
                        df,
                        x='Scan_Type',
                        y='Payload_Size',
                        color='Scan_Type',
                        title="Payload Size Distribution (Box Plot)",
                        color_discrete_map={
                            'Normal': '#2ecc71',
                            'BotAttack': '#e74c3c',
                            'PortScan': '#f39c12'
                        }
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Violin plot for better distribution visualization
                    fig = px.violin(
                        df,
                        x='Scan_Type',
                        y='Payload_Size',
                        color='Scan_Type',
                        title="Payload Size Distribution (Violin Plot)",
                        color_discrete_map={
                            'Normal': '#2ecc71',
                            'BotAttack': '#e74c3c',
                            'PortScan': '#f39c12'
                        }
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Histogram overlay
                fig = px.histogram(
                    df,
                    x='Payload_Size',
                    color='Scan_Type',
                    nbins=50,
                    title="Payload Size Histogram by Scan Type",
                    color_discrete_map={
                        'Normal': '#2ecc71',
                        'BotAttack': '#e74c3c',
                        'PortScan': '#f39c12'
                    },
                    barmode='overlay',
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary
                st.subheader("Payload Size Statistics by Scan Type")
                payload_stats = df.groupby('Scan_Type')['Payload_Size'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
                st.dataframe(payload_stats, use_container_width=True)
                
                st.markdown("""
                **Insight:** Normal traffic has a wide range of payload sizes, while BotAttack traffic 
                typically has larger payloads, and PortScan payloads are limited to smaller sizes. 
                This makes Payload_Size a strong feature for distinguishing malicious activity.
                """)
                
                # Protocol Analysis
                st.subheader("Protocol Distribution by Scan Type")
                if 'Protocol' in df.columns:
                    protocol_cross = pd.crosstab(df['Protocol'], df['Scan_Type'])
                    fig = px.bar(
                        protocol_cross,
                        barmode='group',
                        title="Protocol Usage by Scan Type",
                        labels={'value': 'Count', 'Protocol': 'Protocol'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Status Analysis
                st.subheader("Request Status by Scan Type")
                if 'Status' in df.columns:
                    status_cross = pd.crosstab(df['Status'], df['Scan_Type'])
                    fig = px.bar(
                        status_cross,
                        barmode='group',
                        title="Request Status Distribution by Scan Type",
                        labels={'value': 'Count', 'Status': 'Status'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Insight:** Most Normal requests are successful, while BotAttack and PortScan 
                    requests often fail. This suggests that Status is a strong feature for distinguishing 
                    malicious traffic.
                    """)
        
        with tab4:
            st.header("Target Variable Insights")
            
            if 'Scan_Type' in df.columns:
                # Class balance visualization
                st.subheader("Class Distribution Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    scan_counts = df['Scan_Type'].value_counts()
                    fig = px.pie(
                        values=scan_counts.values,
                        names=scan_counts.index,
                        title="Scan Type Proportion",
                        color_discrete_map={
                            'Normal': '#2ecc71',
                            'BotAttack': '#e74c3c',
                            'PortScan': '#f39c12'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Imbalance Metrics")
                    total = len(df)
                    for scan_type, count in scan_counts.items():
                        pct = (count / total) * 100
                        ratio = count / scan_counts.max()
                        
                        st.metric(
                            label=scan_type,
                            value=f"{count:,}",
                            delta=f"{pct:.1f}% of total"
                        )
                        
                        # Progress bar for visual
                        st.progress(pct / 100)
                
                st.warning("""
                ⚠️ **Class Imbalance Detected:**
                - Normal: 90.4% (majority class)
                - BotAttack: 5.4% (minority class)
                - PortScan: 4.2% (minority class)
                
                This imbalance may require resampling techniques (SMOTE, undersampling, etc.) for better model performance.
                """)
                
                # Intrusion vs Scan Type
                if 'Intrusion' in df.columns:
                    st.subheader("Intrusion Flag Distribution")
                    intrusion_cross = pd.crosstab(df['Intrusion'], df['Scan_Type'])
                    fig = px.bar(
                        intrusion_cross,
                        barmode='group',
                        title="Intrusion Flag by Scan Type",
                        labels={'Intrusion': 'Intrusion (0=Normal, 1=Attack)', 'value': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Results":
    st.markdown('<div class="main-header">Model Training Results</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.markdown("""
        ### Models Evaluated:
        We trained and evaluated multiple classification models:
        1. **Logistic Regression** - Linear classifier
        2. **K-Nearest Neighbors (KNN)** - Instance-based learning
        3. **Decision Tree** - Tree-based classifier
        4. **Random Forest** - Ensemble of decision trees
        5. **XGBoost** - Gradient boosting framework
        6. **Multi-Layer Perceptron (MLP)** - Neural network classifier
        """)
        
        st.markdown("---")
        
        # Model Performance Table
        st.header("📊 Model Performance Comparison")
        
        # Based on results from the notebook
        model_results = {
            'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost', 'MLP'],
            'Accuracy': [0.9992, 0.9778, 0.9992, 0.9996, 0.9996, 0.9970],
            'Precision (Weighted)': [0.9993, 0.9785, 0.9993, 0.9996, 0.9996, 0.9971],
            'Recall (Weighted)': [0.9992, 0.9778, 0.9992, 0.9996, 0.9996, 0.9970],
            'F1-Score (Weighted)': [0.9992, 0.9762, 0.9992, 0.9996, 0.9996, 0.9970],
            'ROC-AUC (Multi-class)': [1.0000, 0.9758, 0.9997, 0.9999, 0.9999, 0.9985]
        }
        
        results_df = pd.DataFrame(model_results)
        
        # Highlight most accurate models
        st.info("""
        📊 **Accuracy Ranking**: 
        - 🥇 **Random Forest & XGBoost**: 99.96% (Most Accurate)
        - 🥈 **Decision Tree & Logistic Regression**: 99.92%
        - 🥉 **MLP**: 99.70%
        - **KNN**: 97.78%
        
        *Note: Decision Tree was selected for interpretability, not maximum accuracy.*
        """)
        
        # Display metrics
        st.dataframe(results_df, use_container_width=True)
        
        # Visualizations
        st.subheader("Performance Metrics Visualization")
        
        tab1, tab2, tab3 = st.tabs(["📊 Accuracy & ROC-AUC", "📈 All Metrics", "🎯 Radar Chart"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    results_df,
                    x='Model',
                    y='Accuracy',
                    title="Model Accuracy Comparison",
                    color='Accuracy',
                    color_continuous_scale="Greens",
                    text='Accuracy'
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    results_df,
                    x='Model',
                    y='ROC-AUC (Multi-class)',
                    title="ROC-AUC Score Comparison",
                    color='ROC-AUC (Multi-class)',
                    color_continuous_scale="Blues",
                    text='ROC-AUC (Multi-class)'
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Create a comparison chart for all metrics
            metrics_to_plot = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 
                              'F1-Score (Weighted)', 'ROC-AUC (Multi-class)']
            
            fig = go.Figure()
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df['Model'],
                    y=results_df[metric],
                    text=results_df[metric].round(4),
                    textposition='outside'
                ))
            
            fig.update_layout(
                title="All Metrics Comparison Across Models",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                height=500,
                xaxis={'tickangle': 45}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Radar chart for top 3 models
            top_3_models = results_df.nlargest(3, 'Accuracy')
            
            metrics_for_radar = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 
                                'F1-Score (Weighted)', 'ROC-AUC (Multi-class)']
            
            fig = go.Figure()
            
            for idx, row in top_3_models.iterrows():
                values = [row[m] for m in metrics_for_radar]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_for_radar,
                    fill='toself',
                    name=row['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0.95, 1.0]
                    )),
                showlegend=True,
                title="Top 3 Models - Radar Chart Comparison",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Best Model Selection
        st.header("🏆 Model Selection: Decision Tree")
        
        st.info("""
        **Note**: While **Random Forest** and **XGBoost** achieved slightly higher accuracy (99.96% vs 99.92%), 
        **Decision Tree** was selected as the final model based on:
        - ✅ **Interpretability**: Easy to understand and explain decisions
        - ✅ **Speed**: Fastest inference time
        - ✅ **Still excellent accuracy**: 99.92% is very high performance
        - ✅ **Feature importance**: Clear feature importance visualization
        - ✅ **Explainability**: Can visualize the decision path
        
        **For production use**, Random Forest or XGBoost would be better choices if accuracy is the primary concern.
        """)
        
        # Detailed metrics for Decision Tree
        st.subheader("Decision Tree Detailed Performance")
        
        dt_metrics = {
            'Metric': ['Precision (BotAttack)', 'Recall (BotAttack)', 'F1 (BotAttack)',
                       'Precision (Normal)', 'Recall (Normal)', 'F1 (Normal)',
                       'Precision (PortScan)', 'Recall (PortScan)', 'F1 (PortScan)'],
            'Score': [1.00, 1.00, 1.00,
                     1.00, 1.00, 1.00,
                     0.98, 1.00, 0.99]
        }
        
        dt_df = pd.DataFrame(dt_metrics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(dt_df, use_container_width=True)
        
        with col2:
            fig = px.bar(
                dt_df,
                x='Metric',
                y='Score',
                title="Decision Tree Per-Class Metrics",
                color='Score',
                color_continuous_scale="Viridis"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Feature Importance (if available)
        st.header("🔍 Feature Importance")
        
        if st.session_state.model is not None:
            try:
                # Get feature names (need to match training features)
                # This is a placeholder - you'd need to save feature names with the model
                feature_names = ['Payload_Size', 'Request_Type_DNS', 'Request_Type_FTP', 
                               'Request_Type_HTTP', 'Request_Type_HTTPS', 'Request_Type_SMTP',
                               'Request_Type_SSH', 'Request_Type_Telnet', 'Protocol_ICMP',
                               'Protocol_TCP', 'Protocol_UDP']
                
                if hasattr(st.session_state.model, 'feature_importances_'):
                    importances = st.session_state.model.feature_importances_
                    # Show top 20 features
                    top_n = min(20, len(importances))
                    top_indices = np.argsort(importances)[-top_n:][::-1]
                    
                    top_features = [f"Feature_{i}" for i in top_indices[:10]]  # Simplified
                    top_importance = importances[top_indices[:10]]
                    
                    fig = px.bar(
                        x=top_importance,
                        y=top_features,
                        orientation='h',
                        title="Top 10 Feature Importances",
                        labels={'x': 'Importance', 'y': 'Feature'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info("Feature importance visualization requires the trained model and feature names.")
        else:
            st.info("💡 Load the model to see feature importance visualization.")

elif page == "🔮 Inference":
    st.markdown('<div class="main-header">Model Inference</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.model is None:
        st.error("⚠️ Model not found. Please ensure 'network_logs_decision_tree_model.joblib' is in the directory.")
        st.info("💡 You can still test the preprocessing pipeline, but predictions won't be available.")
        model_available = False
    else:
        model_available = True
    
    st.markdown("### Input Network Log Features for Classification")
    st.markdown("Fill in the network log details below to classify the traffic type.")
    
    # Import preprocessing function
    try:
        from preprocessing import preprocess_for_inference, create_sample_input
    except ImportError:
        st.error("⚠️ preprocessing.py not found. Please ensure it's in the same directory.")
        st.stop()
    
    # Create input form
    with st.form("inference_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            request_type = st.selectbox(
                "Request Type",
                ["HTTP", "HTTPS", "FTP", "SSH", "SMTP", "DNS", "Telnet"]
            )
            
            protocol = st.selectbox(
                "Protocol",
                ["TCP", "UDP", "ICMP"]
            )
            
            status = st.selectbox(
                "Status",
                ["Success", "Failure"]
            )
            
            port = st.number_input(
                "Port",
                min_value=1,
                max_value=65535,
                value=80,
                help="Common ports: 80 (HTTP), 443 (HTTPS), 22 (SSH), 21 (FTP), etc."
            )
        
        with col2:
            payload_size = st.number_input(
                "Payload Size (bytes)",
                min_value=0,
                value=1000,
                help="Size of the data payload in bytes"
            )
            
            user_agent = st.selectbox(
                "User Agent",
                ["Mozilla/5.0", "Wget/1.20.3", "curl/7.68.0", "nmap/7.80", 
                 "Nikto/2.1.6", "python-requests/2.25.1"],
                help="User agent string identifying the client"
            )
            
            intrusion = st.selectbox(
                "Intrusion Flag",
                [0, 1],
                format_func=lambda x: "Normal (0)" if x == 0 else "Attack (1)",
                help="Note: This feature was dropped during training but can be used for reference"
            )
        
        submitted = st.form_submit_button("🔍 Classify", use_container_width=True)
        
        if submitted:
            try:
                # Create input dictionary
                input_dict = create_sample_input(
                    request_type=request_type,
                    protocol=protocol,
                    status=status,
                    port=port,
                    payload_size=payload_size,
                    user_agent=user_agent,
                    intrusion=intrusion
                )
                
                # Preprocess input
                with st.spinner("Preprocessing input data..."):
                    preprocessed_data = preprocess_for_inference(input_dict)
                
                st.success("✅ Preprocessing Complete!")
                
                # Make prediction if model is available
                if model_available:
                    with st.spinner("Making prediction..."):
                        # Get prediction probabilities
                        probs = st.session_state.model.predict_proba(preprocessed_data)[0]
                        
                        # Label mapping (from training)
                        label_mapping = {0: 'BotAttack', 1: 'Normal', 2: 'PortScan'}
                        predicted_class_idx = np.argmax(probs)
                        predicted_class = label_mapping[predicted_class_idx]
                    
                    st.success("✅ Classification Complete!")
                    
                    # Display probabilities
                    st.markdown("### 📊 Prediction Probabilities")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        botattack_prob = probs[0]
                        st.metric("BotAttack", f"{botattack_prob*100:.2f}%")
                        st.progress(float(botattack_prob))
                        if predicted_class == "BotAttack":
                            st.success("🎯 **Predicted**")
                    
                    with col2:
                        normal_prob = probs[1]
                        st.metric("Normal", f"{normal_prob*100:.2f}%")
                        st.progress(float(normal_prob))
                        if predicted_class == "Normal":
                            st.success("🎯 **Predicted**")
                    
                    with col3:
                        portscan_prob = probs[2]
                        st.metric("PortScan", f"{portscan_prob*100:.2f}%")
                        st.progress(float(portscan_prob))
                        if predicted_class == "PortScan":
                            st.success("🎯 **Predicted**")
                    
                    # Visualize probabilities
                    prob_df = pd.DataFrame({
                        'Class': ['BotAttack', 'Normal', 'PortScan'],
                        'Probability': probs
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='Class',
                        y='Probability',
                        color='Probability',
                        color_continuous_scale="RdYlGn_r",
                        title="Prediction Probabilities",
                        labels={'Probability': 'Probability', 'Class': 'Traffic Type'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    st.header(f"🎯 Predicted Class: **{predicted_class}**")
                    
                    if predicted_class == "Normal":
                        st.success("✅ This traffic appears to be **normal network activity**. No immediate threat detected.")
                    elif predicted_class == "BotAttack":
                        st.error("⚠️ **SECURITY ALERT:** This traffic is classified as **BotAttack**. Automated malicious activity detected. Immediate investigation recommended.")
                        st.warning("**Recommended Actions:** Block source IP, review firewall rules, analyze user agent patterns.")
                    else:  # PortScan
                        st.error("⚠️ **SECURITY ALERT:** This traffic is classified as **PortScan**. Potential reconnaissance activity detected. Investigation recommended.")
                        st.warning("**Recommended Actions:** Monitor source IP, check for repeated connection attempts, review access logs.")
                    
                    # Show input summary
                    with st.expander("📋 Input Summary"):
                        input_summary = pd.DataFrame({
                            'Feature': list(input_dict.keys()),
                            'Value': list(input_dict.values())
                        })
                        st.dataframe(input_summary, use_container_width=True)
                else:
                    st.info("💡 Model not loaded. Preprocessing completed successfully. Here's the preprocessed feature vector:")
                    st.dataframe(preprocessed_data, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error during inference: {str(e)}")
                st.exception(e)
                
                st.info("""
                **Troubleshooting Tips:**
                1. Ensure all input fields are filled correctly
                2. Check that port number is valid (1-65535)
                3. Verify that payload size is a positive number
                4. Make sure preprocessing.py is in the same directory
                """)
    
    st.markdown("---")
    st.markdown("### 📝 Batch Inference")
    
    st.info("""
    **Coming Soon:** Upload a CSV file with multiple network logs for batch classification.
    The CSV should contain columns: Request_Type, Protocol, Status, Port, Payload_Size, User_Agent
    """)

elif page == "📅 Time Series Forecast":
    st.markdown('<div class="main-header">Time Series Forecasting</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Import time series module with error handling
    try:
        import sys
        import warnings
        warnings.filterwarnings('ignore')
        
        # Suppress TensorFlow warnings and set environment before import
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        from time_series_forecasting import (
            load_time_series_data, aggregate_hourly_traffic,
            forecast_arima, forecast_holt_winters, forecast_prophet, forecast_lstm,
            detect_anomalies_zscore, evaluate_forecast, seasonal_decomposition
        )
    except ImportError as e:
        st.error(f"⚠️ Time series forecasting module not found: {str(e)}")
        st.info("Please ensure time_series_forecasting.py is in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading time series module: {str(e)}")
        st.stop()
    
    # Show loading message immediately
    status_placeholder = st.empty()
    status_placeholder.info("⏳ Loading time series data...")
    
    # Load time series data
    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour, no spinner
    def load_ts_data():
        """Load time series data"""
        try:
            return load_time_series_data("Time-Series_Network_logs.csv")
        except Exception as e:
            return None
    
    # Cache aggregated data
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_aggregated_data(ts_data):
        """Aggregate time series data - cached for performance"""
        if ts_data is None:
            return None, None, None
        try:
            return aggregate_hourly_traffic(ts_data)
        except Exception as e:
            return None, None, None
    
    ts_data = load_ts_data()
    status_placeholder.empty()  # Clear loading message
    
    if ts_data is None:
        st.error("⚠️ Error loading time series data. Please ensure 'Time-Series_Network_logs.csv' exists.")
        st.info("💡 The file should have a 'Timestamp' column and 'Intrusion' column.")
        st.stop()
    
    # Aggregate data (cached)
    status_placeholder = st.empty()
    status_placeholder.info("⏳ Processing time series data...")
    hourly_traffic, hourly_malicious, hourly_normal = get_aggregated_data(ts_data)
    status_placeholder.empty()  # Clear loading message
    
    if hourly_traffic is None or len(hourly_traffic) == 0:
        st.error("⚠️ Error aggregating time series data.")
        st.stop()
    
    # Continue if data is loaded successfully
    st.header("📊 Time Series Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Hourly Points", len(hourly_traffic))
    with col2:
        st.metric("Total Traffic", f"{hourly_traffic.sum():,}")
    with col3:
        st.metric("Malicious Traffic", f"{hourly_malicious.sum():,}", 
                 delta=f"{(hourly_malicious.sum()/hourly_traffic.sum()*100):.2f}%")
    
    # Visualization tabs - lazy load to prevent blocking
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Traffic Trends", 
        "🔮 Forecasting", 
        "🔍 Anomaly Detection",
        "📉 Seasonal Decomposition"
    ])
    
    with tab1:
        st.subheader("Hourly Network Traffic Trends")
        
        # Limit data points for faster rendering
        max_points = 500
        if len(hourly_traffic) > max_points:
            # Sample data for display
            step = len(hourly_traffic) // max_points
            display_traffic = hourly_traffic[::step]
            display_malicious = hourly_malicious[::step]
            display_normal = hourly_normal[::step]
        else:
            display_traffic = hourly_traffic
            display_malicious = hourly_malicious
            display_normal = hourly_normal
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=display_traffic.index,
            y=display_traffic.values,
            mode='lines',
            name='Total Traffic',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=display_malicious.index,
            y=display_malicious.values,
            mode='lines',
            name='Malicious Traffic',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=display_normal.index,
            y=display_normal.values,
            mode='lines',
            name='Normal Traffic',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Network Traffic Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Number of Requests",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Traffic Forecasting")
        
        # Forecast parameters
        col1, col2 = st.columns(2)
        with col1:
            forecast_hours = st.slider("Forecast Hours", 1, 48, 24)
        with col2:
            selected_model = st.selectbox(
                "Select Forecasting Model",
                ["ARIMA", "Holt-Winters", "Prophet", "LSTM", "Compare All"]
            )
        
        # Warning for slow models
        if selected_model in ["LSTM", "Compare All"]:
            st.warning("⚠️ LSTM training may take 30-60 seconds. For faster results, use ARIMA or Holt-Winters.")
        
        # Split data (simple, no caching needed - fast operation)
        train_size = int(len(hourly_traffic) * 0.8)
        train = hourly_traffic[:train_size]
        test = hourly_traffic[train_size:train_size+forecast_hours] if len(hourly_traffic) > train_size + forecast_hours else hourly_traffic[train_size:]
        
        if st.button("🔮 Generate Forecast", use_container_width=True):
            forecasts = {}
            forecast_placeholder = st.empty()
            
            # Create forecast index
            last_timestamp = hourly_traffic.index[-1]
            forecast_index = pd.date_range(
                start=last_timestamp + pd.Timedelta(hours=1),
                periods=forecast_hours,
                freq='H'
            )
            
            # Train size for historical display
            train_size_display = int(len(hourly_traffic) * 0.8)
            train_display = hourly_traffic[:train_size_display]
            
            # ARIMA - Fast, show immediately
            if selected_model in ["ARIMA", "Compare All"]:
                with st.spinner("Training ARIMA model..."):
                    arima_forecast, _ = forecast_arima(train, forecast_steps=forecast_hours)
                    if arima_forecast is not None:
                        forecasts['ARIMA'] = arima_forecast
                        # Show immediately
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=train_display.index[-48:],
                            y=train_display.values[-48:],
                            mode='lines',
                            name='Historical Traffic',
                            line=dict(color='blue', width=2)
                        ))
                        colors = {'ARIMA': 'orange', 'Holt-Winters': 'purple', 
                                 'Prophet': 'green', 'LSTM': 'brown'}
                        fig.add_trace(go.Scatter(
                            x=forecast_index,
                            y=arima_forecast,
                            mode='lines',
                            name='ARIMA Forecast',
                            line=dict(color=colors['ARIMA'], width=2, dash='dash')
                        ))
                        fig.update_layout(
                            title="Network Traffic Forecast (ARIMA Ready)",
                            xaxis_title="Timestamp",
                            yaxis_title="Number of Requests",
                            hovermode='x unified',
                            height=500
                        )
                        forecast_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Holt-Winters - Fast, show immediately
            if selected_model in ["Holt-Winters", "Compare All"]:
                with st.spinner("Training Holt-Winters model..."):
                    hw_forecast, _ = forecast_holt_winters(train, forecast_steps=forecast_hours)
                    if hw_forecast is not None:
                        forecasts['Holt-Winters'] = hw_forecast
                        # Update plot with Holt-Winters
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=train_display.index[-48:],
                            y=train_display.values[-48:],
                            mode='lines',
                            name='Historical Traffic',
                            line=dict(color='blue', width=2)
                        ))
                        colors = {'ARIMA': 'orange', 'Holt-Winters': 'purple', 
                                 'Prophet': 'green', 'LSTM': 'brown'}
                        for model_name, forecast_values in forecasts.items():
                            fig.add_trace(go.Scatter(
                                x=forecast_index,
                                y=forecast_values,
                                mode='lines',
                                name=f'{model_name} Forecast',
                                line=dict(color=colors.get(model_name, 'gray'), width=2, dash='dash')
                            ))
                        fig.update_layout(
                            title="Network Traffic Forecast (Updating...)",
                            xaxis_title="Timestamp",
                            yaxis_title="Number of Requests",
                            hovermode='x unified',
                            height=500
                        )
                        forecast_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Prophet - Medium speed, add when ready
            if selected_model in ["Prophet", "Compare All"]:
                with st.spinner("Training Prophet model (this may take 15-30 seconds)..."):
                    # Use a smaller sample for Prophet to avoid hanging
                    prophet_train = train if len(train) <= 200 else train[-200:]
                    prophet_forecast, _ = forecast_prophet(prophet_train, forecast_steps=forecast_hours)
                    if prophet_forecast is not None:
                        # Extract forecast values
                        prophet_values = prophet_forecast['yhat'][-forecast_hours:].values
                        forecasts['Prophet'] = prophet_values
                        # Update plot with Prophet
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=train_display.index[-48:],
                            y=train_display.values[-48:],
                            mode='lines',
                            name='Historical Traffic',
                            line=dict(color='blue', width=2)
                        ))
                        colors = {'ARIMA': 'orange', 'Holt-Winters': 'purple', 
                                 'Prophet': 'green', 'LSTM': 'brown'}
                        for model_name, forecast_values in forecasts.items():
                            fig.add_trace(go.Scatter(
                                x=forecast_index,
                                y=forecast_values,
                                mode='lines',
                                name=f'{model_name} Forecast',
                                line=dict(color=colors.get(model_name, 'gray'), width=2, dash='dash')
                            ))
                        fig.update_layout(
                            title="Network Traffic Forecast (Updating...)",
                            xaxis_title="Timestamp",
                            yaxis_title="Number of Requests",
                            hovermode='x unified',
                            height=500
                        )
                        forecast_placeholder.plotly_chart(fig, use_container_width=True)
            
            # LSTM - Slow, add when ready
            if selected_model in ["LSTM", "Compare All"]:
                with st.spinner("Training LSTM model (this may take 30-60 seconds)..."):
                    lstm_forecast, _ = forecast_lstm(train, forecast_steps=forecast_hours, epochs=15)
                    if lstm_forecast is not None:
                        forecasts['LSTM'] = lstm_forecast
                        # Update plot with LSTM
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=train_display.index[-48:],
                            y=train_display.values[-48:],
                            mode='lines',
                            name='Historical Traffic',
                            line=dict(color='blue', width=2)
                        ))
                        colors = {'ARIMA': 'orange', 'Holt-Winters': 'purple', 
                                 'Prophet': 'green', 'LSTM': 'brown'}
                        for model_name, forecast_values in forecasts.items():
                            fig.add_trace(go.Scatter(
                                x=forecast_index,
                                y=forecast_values,
                                mode='lines',
                                name=f'{model_name} Forecast',
                                line=dict(color=colors.get(model_name, 'gray'), width=2, dash='dash')
                            ))
                        fig.update_layout(
                            title="Network Traffic Forecast (Complete)",
                            xaxis_title="Timestamp",
                            yaxis_title="Number of Requests",
                            hovermode='x unified',
                            height=500
                        )
                        forecast_placeholder.plotly_chart(fig, use_container_width=True)
            
            if forecasts:
                # Final plot update (if not already shown)
                if len(forecasts) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=train_display.index[-48:],
                        y=train_display.values[-48:],
                        mode='lines',
                        name='Historical Traffic',
                        line=dict(color='blue', width=2)
                    ))
                    colors = {'ARIMA': 'orange', 'Holt-Winters': 'purple', 
                             'Prophet': 'green', 'LSTM': 'brown'}
                    for model_name, forecast_values in forecasts.items():
                        fig.add_trace(go.Scatter(
                            x=forecast_index,
                            y=forecast_values,
                            mode='lines',
                            name=f'{model_name} Forecast',
                            line=dict(color=colors.get(model_name, 'gray'), width=2, dash='dash')
                        ))
                    fig.update_layout(
                        title="Network Traffic Forecast",
                        xaxis_title="Timestamp",
                        yaxis_title="Number of Requests",
                        hovermode='x unified',
                        height=500
                    )
                    forecast_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Evaluation metrics - Always show below the chart
                st.markdown("---")
                st.subheader("📊 Forecast Accuracy")
                
                # Calculate metrics for all forecasts
                metrics_df = []
                
                for model_name, forecast_values in forecasts.items():
                    try:
                        # Use test data if available and same length, otherwise use last N values from train
                        if len(test) > 0 and len(test) == len(forecast_values):
                            metrics = evaluate_forecast(test.values, forecast_values)
                        elif len(forecast_values) > 0:
                            # Use last N values from training data for comparison
                            eval_data = train.values[-len(forecast_values):] if len(train) >= len(forecast_values) else train.values
                            forecast_eval = forecast_values[:len(eval_data)]
                            if len(eval_data) == len(forecast_eval):
                                metrics = evaluate_forecast(eval_data, forecast_eval)
                            else:
                                metrics = None
                        else:
                            metrics = None
                        
                        if metrics:
                            metrics_df.append({
                                'Model': model_name,
                                'MAE': round(metrics['MAE'], 2),
                                'RMSE': round(metrics['RMSE'], 2)
                            })
                    except Exception as e:
                        continue
                
                if metrics_df:
                    metrics_df = pd.DataFrame(metrics_df)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Best model
                    best_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
                    best_rmse = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'RMSE']
                    st.success(f"🏆 **Best Model: {best_model}** (Lowest RMSE: {best_rmse:.2f})")
                else:
                    st.info("💡 Forecast accuracy metrics will be calculated when test data is available.")
            else:
                st.warning("⚠️ No forecasts could be generated. Please check model dependencies.")
    
    with tab3:
        st.subheader("Anomaly Detection")
        
        threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 2.0, 0.5)
        
        if st.button("🔍 Detect Anomalies", use_container_width=True):
            with st.spinner("Detecting anomalies..."):
                anomalies = detect_anomalies_zscore(hourly_traffic, threshold=threshold)
            
            st.metric("Anomalies Detected", len(anomalies))
            
            # Plot with anomalies
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hourly_traffic.index,
                y=hourly_traffic.values,
                mode='lines',
                name='Total Traffic',
                line=dict(color='blue', width=1)
            ))
            
            if len(anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=anomalies.index,
                    y=anomalies.values,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10, symbol='x')
                ))
            
            fig.update_layout(
                title=f"Anomaly Detection (Threshold: {threshold})",
                xaxis_title="Timestamp",
                yaxis_title="Number of Requests",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly statistics
            if len(anomalies) > 0:
                st.subheader("Anomaly Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Anomaly Count", len(anomalies))
                    st.metric("Anomaly Percentage", f"{(len(anomalies)/len(hourly_traffic)*100):.2f}%")
                with col2:
                    st.metric("Max Anomaly Value", int(anomalies.max()))
                    st.metric("Min Anomaly Value", int(anomalies.min()))
    
    with tab4:
        st.subheader("Seasonal Decomposition")
        
        if st.button("📉 Decompose Time Series", use_container_width=True):
            with st.spinner("Decomposing time series..."):
                decomposition = seasonal_decomposition(hourly_traffic, period=24)
            
            if decomposition is not None:
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                    vertical_spacing=0.1
                )
                
                # Original
                fig.add_trace(
                    go.Scatter(x=hourly_traffic.index, y=hourly_traffic.values, name='Original'),
                    row=1, col=1
                )
                
                # Trend
                fig.add_trace(
                    go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name='Trend'),
                    row=2, col=1
                )
                
                # Seasonal
                fig.add_trace(
                    go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name='Seasonal'),
                    row=3, col=1
                )
                
                # Residual
                fig.add_trace(
                    go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name='Residual'),
                    row=4, col=1
                )
                
                fig.update_layout(height=800, showlegend=False)
                fig.update_xaxes(title_text="Timestamp", row=4, col=1)
                fig.update_yaxes(title_text="Value", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not perform decomposition. Check data requirements.")

else:  # About page
    st.markdown('<div class="main-header">About This Project</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ## 🛡️ Network Intrusion Detection System
    
    This project focuses on detecting different types of network activity—specifically distinguishing 
    between **Normal traffic**, **Bot Attacks**, and **Port Scan Attacks**—based on structured network logs.
    
    ### 🎯 Objective
    
    Using the `Scan_Type` column as our target label, we aim to build a multiclass classification 
    model that can identify the nature of each network request. This task simulates a real-world 
    intrusion detection system (IDS) that goes beyond simple anomaly detection.
    
    ### 📊 Dataset Features
    
    - **Source_IP / Destination_IP**: IP addresses involved in the connection
    - **Port**: Network port number
    - **Request_Type**: Type of request (HTTP, HTTPS, FTP, SSH, etc.)
    - **Protocol**: Network protocol (TCP, UDP, ICMP)
    - **Payload_Size**: Size of data payload in bytes
    - **User_Agent**: Client user agent string
    - **Status**: Request status (Success/Failure)
    - **Intrusion**: Binary target (0 = Normal, 1 = Attack)
    - **Scan_Type**: Traffic type classification (Normal, PortScan, BotAttack)
    
    ### 🤖 Models Used
    
    1. **Logistic Regression** - Linear classifier with regularization
    2. **K-Nearest Neighbors** - Instance-based learning algorithm
    3. **Decision Tree** - Interpretable tree-based classifier ⭐ (Selected)
    4. **Random Forest** - Ensemble of decision trees
    5. **XGBoost** - Gradient boosting framework
    6. **Multi-Layer Perceptron** - Neural network classifier
    
    ### 📈 Key Findings
    
    - **Class Imbalance**: Normal traffic dominates (90.4%), while attack types are underrepresented
    - **High Performance**: All models achieved >97% accuracy
    - **Selected Model**: Decision Tree with 99.92% accuracy (chosen for interpretability)
    - **Most Accurate Models**: Random Forest & XGBoost with 99.96% accuracy
    - **Feature Importance**: Payload size, Status, and Protocol are key discriminators
    
    ### 🛠️ Technology Stack
    
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: Scikit-learn, XGBoost
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Deployment**: Streamlit
    - **Model Explanation**: SHAP values
    
    ### 📝 Notes
    
    - The Intrusion feature was dropped during training to encourage better generalization
    - One-hot encoding was applied to categorical features
    - Payload size was standardized using StandardScaler
    - Stratified train-test split (70-30) was used to maintain class distribution
    """)

if __name__ == "__main__":
    pass

