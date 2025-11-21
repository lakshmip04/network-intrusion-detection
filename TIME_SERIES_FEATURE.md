# 📅 Time Series Forecasting Feature - Documentation

## ✅ What Was Added

A comprehensive **Time Series Forecasting** page has been added to your Streamlit application!

## 🎯 Features

### 1. **Traffic Trends Visualization**
- Interactive plot showing total, malicious, and normal traffic over time
- Hourly aggregated data
- Real-time hover information

### 2. **Forecasting Models**
Four different forecasting models available:

#### **ARIMA** (AutoRegressive Integrated Moving Average)
- Best for: Short-term forecasts
- Parameters: (1,1,1) order
- Fast training and prediction

#### **Holt-Winters Exponential Smoothing**
- Best for: Data with trends and seasonality
- Captures hourly patterns (24-hour seasonality)
- Good for capturing seasonal patterns

#### **Prophet** (Facebook's Prophet)
- Best for: Robust forecasting with holidays/events
- Handles missing data well
- Automatic seasonality detection

#### **LSTM** (Long Short-Term Memory)
- Best for: Complex patterns and long-term dependencies
- Deep learning approach
- Requires more training time

### 3. **Anomaly Detection**
- Z-score based anomaly detection
- Configurable threshold (1.0 to 4.0)
- Visual identification of unusual traffic patterns
- Statistics on detected anomalies

### 4. **Seasonal Decomposition**
- Decomposes time series into:
  - **Trend**: Long-term direction
  - **Seasonal**: Repeating patterns
  - **Residual**: Random noise
- Helps understand underlying patterns

## 📊 How to Use

### Step 1: Access the Page
1. Start your Streamlit app: `streamlit run streamlit_app.py`
2. Navigate to **"📅 Time Series Forecast"** in the sidebar

### Step 2: View Traffic Trends
- Go to the **"📈 Traffic Trends"** tab
- See historical traffic patterns
- Compare total, malicious, and normal traffic

### Step 3: Generate Forecasts
1. Go to **"🔮 Forecasting"** tab
2. Select forecast hours (1-48)
3. Choose a model:
   - Single model: ARIMA, Holt-Winters, Prophet, or LSTM
   - Compare All: See all models side-by-side
4. Click **"🔮 Generate Forecast"**
5. View forecast visualization and accuracy metrics

### Step 4: Detect Anomalies
1. Go to **"🔍 Anomaly Detection"** tab
2. Adjust Z-score threshold (default: 2.0)
3. Click **"🔍 Detect Anomalies"**
4. View detected anomalies on the plot
5. Check anomaly statistics

### Step 5: Seasonal Analysis
1. Go to **"📉 Seasonal Decomposition"** tab
2. Click **"📉 Decompose Time Series"**
3. View trend, seasonal, and residual components

## 📁 Files Added/Modified

### New Files
- `time_series_forecasting.py` - Time series forecasting module

### Modified Files
- `streamlit_app.py` - Added new "Time Series Forecast" page

## 🔧 Dependencies

The time series feature requires:
- ✅ `statsmodels` - For ARIMA and Holt-Winters
- ✅ `prophet` - For Prophet forecasting
- ✅ `tensorflow` - For LSTM (optional)
- ✅ `scikit-learn` - For metrics and scaling

All dependencies are already in your `requirements.txt`!

## 📊 Data Requirements

The time series feature uses:
- **File**: `Time-Series_Network_logs.csv`
- **Required Columns**:
  - `Timestamp` - DateTime column
  - `Intrusion` - Binary flag (0 or 1)
  - Other network log columns

## 🎨 Visualizations

### Traffic Trends
- Line chart with multiple series
- Color-coded: Blue (Total), Red (Malicious), Green (Normal)
- Interactive hover tooltips

### Forecasts
- Historical data (last 48 hours)
- Forecasted values (dashed lines)
- Different colors for each model
- Forecast accuracy metrics table

### Anomalies
- Traffic line with anomaly markers
- Red X markers for anomalies
- Anomaly statistics panel

### Decomposition
- 4-panel subplot
- Original, Trend, Seasonal, Residual
- Time-aligned x-axis

## ⚙️ Configuration

### Forecast Parameters
- **Forecast Hours**: 1-48 hours ahead
- **ARIMA Order**: (1,1,1) - can be modified in code
- **Seasonal Period**: 24 hours (hourly data)
- **LSTM Epochs**: 30 (reduced for faster training)

### Anomaly Detection
- **Z-Score Threshold**: 1.0 to 4.0
- **Rolling Window**: 24 hours
- **Default Threshold**: 2.0 (2 standard deviations)

## 🚀 Performance Tips

1. **For Quick Forecasts**: Use ARIMA or Holt-Winters
2. **For Best Accuracy**: Use Prophet or LSTM
3. **For Comparison**: Use "Compare All" option
4. **LSTM Training**: May take 1-2 minutes (30 epochs)

## 📈 Model Comparison

Based on your notebook results:
- **Best Model**: ARIMA (Lowest RMSE: 7.14)
- **Second Best**: LSTM (RMSE: 7.16)
- **Third**: Prophet (RMSE: 8.12)
- **Fourth**: Holt-Winters (RMSE: 8.96)

## 🎯 Use Cases

1. **Capacity Planning**: Forecast future traffic loads
2. **Anomaly Detection**: Identify unusual traffic spikes
3. **Pattern Analysis**: Understand seasonal trends
4. **Security Monitoring**: Predict malicious traffic patterns
5. **Resource Allocation**: Plan for traffic increases

## 🔍 Troubleshooting

### Issue: "Time series data not available"
**Solution**: Ensure `Time-Series_Network_logs.csv` is in the project directory

### Issue: "Model not available"
**Solution**: 
- Check if required packages are installed: `pip install -r requirements.txt`
- Some models (LSTM) require TensorFlow which may have dependencies

### Issue: "Forecast takes too long"
**Solution**: 
- Use ARIMA or Holt-Winters for faster results
- Reduce LSTM epochs in the code
- Use fewer forecast hours

### Issue: "No anomalies detected"
**Solution**: 
- Lower the Z-score threshold
- Check if data has sufficient variance

## 📝 Notes

- The time series data is aggregated hourly
- Forecasts are generated for future hours
- Anomaly detection uses rolling statistics
- All visualizations are interactive (Plotly)

## 🎉 Ready to Use!

Your Streamlit app now has a complete time series forecasting feature. Just restart the app and navigate to the new page!

```bash
streamlit run streamlit_app.py
```

Then click on **"📅 Time Series Forecast"** in the sidebar!


