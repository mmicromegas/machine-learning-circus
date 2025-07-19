import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import xgboost as xgb

#from keras.models import Sequential  # github copilot : IDE shows error but code works :|
#from keras.layers import LSTM, Dense # github copilot # IDE shows error but code works :|

# miro fix
import tensorflow as tf
import keras
from keras import layers
from keras import models


# Read the CSV file
df = pd.read_csv('DATA/annual-working-hours-per-worker.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
# make the default selection in dropdown to be Slovakia
app.layout = html.Div([
     html.H1("Average Annual and Daily Working Hours per Worker"),
     dcc.Dropdown(
         id='entity-dropdown',
         options=[{'label': entity, 'value': entity} for entity in df['Entity'].unique()],
         value=['Slovakia'],
         multi=True
     ),
     dcc.Graph(id='daily-working-hours-graph')
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

# Callback to update the graphs based on selected entities
@app.callback(
    Output('daily-working-hours-graph', 'figure'),
    [Input('entity-dropdown', 'value')]
)
def update_graphs(selected_entities):
    filtered_df = df[df['Entity'].isin(selected_entities)]

    # Forecasting function
    def forecast_expsmooth(df, column, periods=100):
        model = ExponentialSmoothing(df[column], trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(periods)
        return forecast

    # Forecasting function using XGBoost
    def forecast_xgboost(df, column, periods=100, n_steps=3):
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
    def forecast_lstm(df, column, periods=100, n_steps=3):
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

    # Calculate average daily working hours per worker
    filtered_df['Average daily working hours per worker'] = filtered_df[
                                                                'Average annual working hours per worker'] / 365.24

    # Plot for average daily working hours per worker
    fig_daily = px.line(filtered_df, x='Year', y='Average daily working hours per worker', color='Entity',
                        title='Average Daily Working Hours per Worker')

    # Forecast for average daily working hours per worker
    for entity in selected_entities:
        entity_df = filtered_df[filtered_df['Entity'] == entity]
        forecast_daily_expsmooth = forecast_expsmooth(entity_df, 'Average daily working hours per worker')
        forecast_daily_xgboost = forecast_xgboost(entity_df, 'Average daily working hours per worker')
        forecast_daily_lstm = forecast_lstm(entity_df, 'Average daily working hours per worker')

        forecast_years_expsmooth = list(range(entity_df['Year'].max() + 1, entity_df['Year'].max() + 1 + len(forecast_daily_expsmooth)))
        forecast_years_xgboost = list(range(entity_df['Year'].max() + 1, entity_df['Year'].max() + 1 + len(forecast_daily_xgboost)))
        forecast_years_lstm = list(range(entity_df['Year'].max() + 1, entity_df['Year'].max() + 1 + len(forecast_daily_lstm)))

        fig_daily.add_scatter(x=forecast_years_expsmooth, y=forecast_daily_expsmooth, mode='lines', name=f'{entity} Forecast ExpSmooth')
        fig_daily.add_scatter(x=forecast_years_xgboost, y=forecast_daily_xgboost, mode='lines', name=f'{entity} Forecast Xgboost')
        fig_daily.add_scatter(x=forecast_years_lstm, y=forecast_daily_lstm, mode='lines', name=f'{entity} Forecast LSTM')



    return fig_daily


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)