import sys
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
import xgboost as xgb
import tensorflow as tf
import keras
from keras import layers, models

# Read the CSV file
df = pd.read_csv('../DATA/monthly_close_5_years.csv')

# Ensure the Date column is in datetime format and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Check for and remove any duplicate labels in the date index
df = df[~df.index.duplicated(keep='first')]

# Check the frequency of the date index and set it explicitly if not inferred
if df.index.freq is None:
    df = df.asfreq('M')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Cryptocurrency \"monthly\" Close"),
    dcc.Dropdown(
        id='entity-dropdown',
        options=[{'label': entity, 'value': entity} for entity in df['Cryptocurrency'].unique()],
        value=['Bitcoin'],
        multi=True
    ),
    dcc.Graph(id='bitcoin-monthly-graph'),
    dcc.Graph(id='stl-decomposition-graph')
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
    [Output('bitcoin-monthly-graph', 'figure'),
     Output('stl-decomposition-graph', 'figure')],
    [Input('entity-dropdown', 'value')]
)
def update_graphs(selected_entities):
    filtered_df = df[df['Cryptocurrency'].isin(selected_entities)]

    # Forecasting function
    def forecast_expsmooth(df, column, periods=10):
        model = ExponentialSmoothing(df[column], trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(periods)
        return forecast

    # Forecasting function using XGBoost
    def forecast_xgboost(df, column, periods=10, n_steps=3):
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
    def forecast_lstm(df, column, periods=10, n_steps=3):
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

    # STL Decomposition function
    def stl_decomposition(df, column):
        stl = STL(df[column], seasonal=13)
        result = stl.fit()
        return result

    # Plot for Close_Price
    fig_daily = px.line(filtered_df, x=filtered_df.index, y='Close_Price', color='Cryptocurrency',
                        title='Close_Price (USD)')

    # Forecast for Close_Price
    for entity in selected_entities:
        entity_df = filtered_df[filtered_df['Cryptocurrency'] == entity]

        forecast_daily_expsmooth = forecast_expsmooth(entity_df, 'Close_Price')
        forecast_daily_xgboost = forecast_xgboost(entity_df, 'Close_Price')
        forecast_daily_lstm = forecast_lstm(entity_df, 'Close_Price')

        # Create list of dates for forecast
        last_date = entity_df.index.max()
        forecast_dates_expsmooth = [last_date + pd.DateOffset(months=i) for i in range(1, 1 + len(forecast_daily_expsmooth))]
        forecast_dates_xgboost = [last_date + pd.DateOffset(months=i) for i in range(1, 1 + len(forecast_daily_xgboost))]
        forecast_dates_lstm = [last_date + pd.DateOffset(months=i) for i in range(1, 1 + len(forecast_daily_lstm))]

        fig_daily.add_scatter(x=forecast_dates_expsmooth, y=forecast_daily_expsmooth, mode='lines', name=f'{entity} Forecast ExpSmooth')
        fig_daily.add_scatter(x=forecast_dates_xgboost, y=forecast_daily_xgboost, mode='lines', name=f'{entity} Forecast Xgboost')
        fig_daily.add_scatter(x=forecast_dates_lstm, y=forecast_daily_lstm, mode='lines', name=f'{entity} Forecast LSTM')

    # STL Decomposition plots
    stl_figures = []
    for entity in selected_entities:
        entity_df = filtered_df[filtered_df['Cryptocurrency'] == entity]
        stl_result = stl_decomposition(entity_df, 'Close_Price')

        stl_fig = px.line(title=f'STL Decomposition for {entity}')
        stl_fig.add_scatter(x=entity_df.index, y=stl_result.trend, mode='lines', name='Trend')
        stl_fig.add_scatter(x=entity_df.index, y=stl_result.seasonal, mode='lines', name='Seasonal')
        stl_fig.add_scatter(x=entity_df.index, y=stl_result.resid, mode='lines', name='Residual')
        stl_figures.append(stl_fig)

    return fig_daily, stl_figures[0] if stl_figures else {}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)