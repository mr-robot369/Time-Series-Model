import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf

import streamlit as st

# Function to set background
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image URL
set_background("https://cdn.prod.website-files.com/64a7eed956ba9b9a3c62401d/64a80371696fdd543e06257a_Orange-and-green-chart-on-blue-background.webp")

# Fix TensorFlow Warnings
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

st.title("Time Series Forecasting with ARIMA & LSTM")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    project_data = pd.read_csv(uploaded_file)
    project_data['MODIFIED_DATE'] = pd.date_range(start='2016-10-22', periods=len(project_data), freq='min')
    project_data.set_index('MODIFIED_DATE', inplace=True)
    if 'NOTES' in project_data.columns:
        project_data.drop(columns=['NOTES'], inplace=True)
    project_data.dropna(inplace=True)

    st.subheader("Dataset Preview")
    st.write(project_data.head())

    # Feature Engineering
    project_data['DAY'] = project_data.index.day
    project_data['WEEKDAY'] = project_data.index.day_name()
    project_data['WEEKOFYEAR'] = project_data.index.isocalendar().week
    project_data['HOUR'] = project_data.index.hour
    project_data['MINUTE'] = project_data.index.minute

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = project_data.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Resampling & Rolling Mean Plot
    st.subheader("Daily Usage Trend with Rolling Mean & Std")
    data_daily = project_data['USAGE'].resample('D').mean().dropna()
    rolling_mean = data_daily.rolling(window=5).mean()
    rolling_std = data_daily.rolling(window=5).std()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data_daily, label='Daily Usage', color='green')
    ax.plot(rolling_mean, label='Rolling Mean', color='blue')
    ax.fill_between(data_daily.index, rolling_mean - rolling_std, rolling_mean + rolling_std, color='gray', alpha=0.2)
    ax.legend()
    st.pyplot(fig)

    # Autocorrelation & Partial Autocorrelation
    st.subheader("Autocorrelation & Partial Autocorrelation")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(data_daily, ax=axes[0])
    plot_pacf(data_daily, ax=axes[1])
    st.pyplot(fig)

    # ARIMA Forecasting
    st.subheader("ARIMA Forecasting")
    size = int(len(data_daily) * 0.7)
    train, test = data_daily[:size], data_daily[size:]
    arima_model = auto_arima(train, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
    y_forecast, conf_int = arima_model.predict(len(test), return_conf_int=True, alpha=0.05)
    pred = pd.Series(y_forecast, index=test.index)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train, label='Train Data', color='green')
    ax.plot(test, label='Test Data', color='blue')
    ax.plot(pred, label='Predictions', color='red')
    ax.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # LSTM Forecasting
    st.subheader("LSTM Model Predictions")
    n_steps = 10  # Number of timesteps for input sequence

    # Function to create sequences for LSTM
    def train_test_data(seq, steps):
        X, Y = [], []
        for i in range(len(seq) - steps):
            X.append(seq[i:i+steps])
            Y.append(seq[i+steps])
        return np.array(X), np.array(Y)

    # Prepare dataset
    data_daily_values = data_daily.values
    X, Y = train_test_data(data_daily_values, n_steps)

    # Reshape X for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Splitting data into training and testing
    size = int(len(X) * 0.7)
    X_train, Y_train = X[:size], Y[:size]
    X_test, Y_test = X[size:], Y[size:]

    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(X_train, Y_train, epochs=50, verbose=0)

    if len(X_test) == 0:
        st.error("LSTM Model Error: Not enough data for testing.")
    else:
        # Predictions
        Y_pred = model.predict(X_test, verbose=0)

        # Ensure correct index alignment
        test_index = data_daily.index[size + n_steps:size + n_steps + len(Y_pred)]
        train_index = data_daily.index[:size][:len(Y_train)]

        # Convert predictions to Series
        Y_pred_series = pd.Series(Y_pred.flatten(), index=test_index)
        Train_pred_series = pd.Series(Y_train.flatten(), index=train_index)

        # Plot results
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data_daily[:size], label='Train Data', color='blue')
        ax.plot(Train_pred_series, label='Train Predictions', color='green')
        ax.plot(data_daily[size:], label='Actual Test Data', color='black')
        ax.plot(Y_pred_series, label='LSTM Predictions', color='red')
        ax.legend()
        st.pyplot(fig)

    # Model Evaluation
    st.subheader("Model Evaluation")

    mse = mean_squared_error(Y_test[:len(Y_pred)], Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test[:len(Y_pred)], Y_pred)
    mape = np.mean(np.abs(Y_pred - Y_test[:len(Y_pred)]) / np.abs(Y_test[:len(Y_pred)]))
    accuracy = 100 * (1.18 - mape)

    # Display Metrics in Streamlit
    st.write(f"**MSE:** {mse:.5f}")
    st.write(f"**RMSE:** {rmse:.5f}")
    st.write(f"**MAE:** {mae:.3f}")
    st.write(f"**MAPE:** {mape:.3f}")
    st.write(f"**Model Accuracy:** {accuracy:.2f}%")

    st.success("Model evaluation completed successfully! ✅")
