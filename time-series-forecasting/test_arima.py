import sys
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import tensorflow as tf
import keras
from keras import layers
from keras import models

# Read the CSV file
df = pd.read_csv('DATA/bitcoin_monthly_close_updated.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
     html.H1("Cryptocurrency \"monthly\" Close"),
     dcc.Dropdown(
         id='entity-dropdown',
         options=[{'label': entity, 'value': entity} for entity in df['Currency'].unique()],
         value=['Bitcoin'],
         multi=True
     ),
     dcc.Graph(id='currency-monthly-graphs')
])

# Function to preprocess data for XGBoost
def preprocess_data(data, n_steps=3):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(data_scaled) - n_steps):
        X.append(data_scaled[i:i + n_steps].flatten())
        y.append(data_scaled[i + n_steps])
    return np.array(X), np.array(y), scaler

# Function to build and train XGBoost model
def build_and_train_xgboost(X, y):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(X, y)
    return model

# Function to build and train LSTM model
def build_and_train_lstm(X, y, n_steps=3, n_features=1):
    model = models.Sequential()
    model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    return model

# Function to build and train ARIMA model
def build_and_train_arima(data, order):
    model = ARIMA(data, order=order)
    fit = model.fit()
    return fit

# Callback to update the graphs based on selected entities
@app.callback(
    Output('currency-monthly-graphs', 'figure'),
    [Input('entity-dropdown', 'value')]
)
def update_graphs(selected_entities):
    filtered_df = df[df['Currency'].isin(selected_entities)]

    # Forecasting function
    def forecast_expsmooth(entity_df, periods):
        data = entity_df.set_index('Year_Month').drop(columns=['Currency'])
        model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(periods)
        return forecast

    # Forecasting function using XGBoost
    def forecast_xgboost(df, column, periods, n_steps=3):
        data = df[[column]].values
        X, y, scaler = preprocess_data(data, n_steps)
        model = build_and_train_xgboost(X, y)
        forecast = []
        input_seq = data[-n_steps:]
        for _ in range(periods):
            input_seq_scaled = scaler.transform(input_seq)
            input_seq_scaled = input_seq_scaled.flatten().reshape(1, -1)
            yhat = model.predict(input_seq_scaled)
            yhat = scaler.inverse_transform(yhat.reshape(-1, 1))
            forecast.append(yhat[0, 0])
            input_seq = np.append(input_seq[1:], yhat, axis=0)
        return forecast

    # Forecasting function using LSTM
    def forecast_lstm(df, column, periods, n_steps=3):
        data = df[[column]].values
        X, y, scaler = preprocess_data(data, n_steps)
        model = build_and_train_lstm(X, y, n_steps, n_features=1)
        forecast = []
        input_seq = data[-n_steps:]
        for _ in range(periods):
            input_seq_scaled = scaler.transform(input_seq)
            input_seq_scaled = input_seq_scaled.reshape((1, n_steps, 1))
            yhat = model.predict(input_seq_scaled, verbose=0)
            yhat = scaler.inverse_transform(yhat)
            forecast.append(yhat[0, 0])
            input_seq = np.append(input_seq[1:], yhat, axis=0)
        return forecast

    # Forecasting function using ARIMA
    def forecast_arima(df, column, periods, order=(24, 1, 6)):
        data = df[column].values
        model = build_and_train_arima(data, order)
        forecast = model.forecast(steps=periods)
        return forecast

    # Plot for Average_Close
    fig_daily = px.line(filtered_df, x='Year_Month', y='Average_Close', color='Currency',
                        title='Average_Close (USD)')

    # Forecast for Average_Close
    for entity in selected_entities:
        entity_df = filtered_df[filtered_df['Currency'] == entity]

        train_size = int(len(entity_df) * 0.8)
        train, test = entity_df[:train_size], entity_df[train_size:]

        periods = len(test)

        forecast_daily_expsmooth = forecast_expsmooth(train, periods)
        forecast_daily_xgboost = forecast_xgboost(train, 'Average_Close', periods)
        forecast_daily_lstm = forecast_lstm(train, 'Average_Close', periods)
        forecast_daily_arima = forecast_arima(train, 'Average_Close', periods)

        # Create list of dates for forecast
        last_date = pd.to_datetime(train['Year_Month'].max())
        forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 1 + periods)]

        fig_daily.add_scatter(x=forecast_dates, y=forecast_daily_expsmooth, mode='lines', name=f'{entity} Forecast ExpSmooth')
        fig_daily.add_scatter(x=forecast_dates, y=forecast_daily_xgboost, mode='lines', name=f'{entity} Forecast Xgboost')
        fig_daily.add_scatter(x=forecast_dates, y=forecast_daily_lstm, mode='lines', name=f'{entity} Forecast LSTM')
        fig_daily.add_scatter(x=forecast_dates, y=forecast_daily_arima, mode='lines', name=f'{entity} Forecast ARIMA')

        # Plot test data
        fig_daily.add_scatter(x=test['Year_Month'], y=test['Average_Close'], mode='lines', name=f'{entity} Test Data')

    return fig_daily

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)