"""
Time Series Forecasting Module for Network Traffic Analysis

This module provides time series forecasting capabilities for network traffic data.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Defer TensorFlow import to avoid blocking
TENSORFLOW_AVAILABLE = False
def _import_tensorflow():
    """Lazy import TensorFlow to avoid blocking"""
    global TENSORFLOW_AVAILABLE
    if not TENSORFLOW_AVAILABLE:
        try:
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            TENSORFLOW_AVAILABLE = True
            return True
        except ImportError:
            TENSORFLOW_AVAILABLE = False
            return False
    return True


def load_time_series_data(file_path="Time-Series_Network_logs.csv"):
    """
    Load and prepare time series data.
    
    Args:
        file_path: Path to the time series CSV file
    
    Returns:
        DataFrame with Timestamp as index
    """
    try:
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"Error loading time series data: {str(e)}")


def aggregate_hourly_traffic(df):
    """
    Aggregate network traffic data by hour.
    
    Args:
        df: DataFrame with Timestamp index
    
    Returns:
        Tuple of (total_traffic, malicious_traffic, normal_traffic)
    """
    hourly_traffic = df.resample('H').size().asfreq('H', fill_value=0)
    hourly_malicious = df[df['Intrusion'] == 1].resample('H').size().asfreq('H', fill_value=0)
    hourly_normal = df[df['Intrusion'] == 0].resample('H').size().asfreq('H', fill_value=0)
    
    return hourly_traffic, hourly_malicious, hourly_normal


def forecast_arima(train_data, forecast_steps=24, order=(1, 1, 1)):
    """
    Forecast using ARIMA model.
    
    Args:
        train_data: Training time series data
        forecast_steps: Number of steps to forecast
        order: ARIMA order (p, d, q)
    
    Returns:
        Forecast values and model
    """
    if not STATSMODELS_AVAILABLE:
        return None, None
    
    try:
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=forecast_steps)
        return forecast, fitted_model
    except Exception as e:
        return None, None


def forecast_holt_winters(train_data, forecast_steps=24, seasonal_periods=24):
    """
    Forecast using Holt-Winters Exponential Smoothing.
    
    Args:
        train_data: Training time series data
        forecast_steps: Number of steps to forecast
        seasonal_periods: Seasonal period (default 24 for hourly data)
    
    Returns:
        Forecast values and model
    """
    if not STATSMODELS_AVAILABLE:
        return None, None
    
    try:
        model = ExponentialSmoothing(
            train_data, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=seasonal_periods
        )
        fitted_model = model.fit()
        forecast = fitted_model.forecast(forecast_steps)
        return forecast, fitted_model
    except Exception as e:
        return None, None


def forecast_prophet(train_data, forecast_steps=24):
    """
    Forecast using Facebook Prophet.
    
    Args:
        train_data: Training time series data (Series with DatetimeIndex)
        forecast_steps: Number of steps to forecast
    
    Returns:
        Forecast DataFrame and model
    """
    if not PROPHET_AVAILABLE:
        return None, None
    
    try:
        # Convert to Prophet format
        prophet_df = train_data.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Optimize Prophet for speed
        model = Prophet(
            daily_seasonality=True, 
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,  # Faster convergence
            mcmc_samples=0  # Disable MCMC for speed
        )
        model.fit(prophet_df, verbose=False)  # Disable verbose output
        
        future = model.make_future_dataframe(periods=forecast_steps, freq='H')
        forecast = model.predict(future)
        
        return forecast, model
    except Exception as e:
        return None, None


def forecast_lstm(train_data, forecast_steps=24, seq_length=24, epochs=20):
    """
    Forecast using LSTM neural network.
    
    Args:
        train_data: Training time series data
        forecast_steps: Number of steps to forecast
        seq_length: Sequence length for LSTM
        epochs: Number of training epochs (reduced for speed)
    
    Returns:
        Forecast values and scaler
    """
    if not SKLEARN_AVAILABLE:
        return None, None
    
    # Lazy import TensorFlow to avoid blocking on page load
    try:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
    except ImportError:
        return None, None
    except Exception as e:
        # If TensorFlow has issues, just return None
        return None, None
    
    try:
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
        
        # Create sequences
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(scaled_data, seq_length)
        if len(X) == 0:
            return None, None
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build LSTM model (simplified for speed)
        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(seq_length, 1)))  # Reduced units
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # Train with reduced epochs and batch size for speed
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        
        # Forecast
        last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
        forecasts = []
        
        for _ in range(forecast_steps):
            pred = model.predict(last_sequence, verbose=0)
            forecasts.append(pred[0, 0])
            # Update sequence
            last_sequence = np.append(last_sequence[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
        
        # Inverse transform
        forecast_array = np.array(forecasts).reshape(-1, 1)
        forecast = scaler.inverse_transform(forecast_array).flatten()
        
        return forecast, scaler
    except Exception as e:
        return None, None


def detect_anomalies_zscore(data, threshold=2):
    """
    Detect anomalies using Z-score method.
    
    Args:
        data: Time series data
        threshold: Z-score threshold
    
    Returns:
        Series of anomaly indices
    """
    rolling_mean = data.rolling(24).mean()
    rolling_std = data.rolling(24).std()
    z_score = (data - rolling_mean) / rolling_std
    anomalies = data[z_score.abs() > threshold]
    return anomalies


def evaluate_forecast(true_values, predicted_values):
    """
    Evaluate forecast accuracy.
    
    Args:
        true_values: Actual values
        predicted_values: Predicted values
    
    Returns:
        Dictionary with MAE, MSE, RMSE
    """
    if not SKLEARN_AVAILABLE:
        return None
    
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }


def seasonal_decomposition(data, period=24):
    """
    Perform seasonal decomposition of time series.
    
    Args:
        data: Time series data
        period: Seasonal period
    
    Returns:
        Decomposition result
    """
    if not STATSMODELS_AVAILABLE:
        return None
    
    try:
        decomposition = seasonal_decompose(data, model='additive', period=period)
        return decomposition
    except Exception as e:
        return None

