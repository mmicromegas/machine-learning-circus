import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import xgboost as xgb

# Read the CSV file
df = pd.read_csv('../DATA/annual-working-hours-per-worker.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Average Annual and Daily Working Hours per Worker (XGBoost)"),
    dcc.Dropdown(
        id='entity-dropdown',
        options=[{'label': entity, 'value': entity} for entity in df['Entity'].unique()],
        value=[df['Entity'].unique()[0]],
        multi=True
    ),
    dcc.Graph(id='annual-working-hours-graph'),
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

# Callback to update the graphs based on selected entities
@app.callback(
    [Output('annual-working-hours-graph', 'figure'),
     Output('daily-working-hours-graph', 'figure')],
    [Input('entity-dropdown', 'value')]
)
def update_graphs(selected_entities):
    filtered_df = df[df['Entity'].isin(selected_entities)]

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

    # Plot for average annual working hours per worker
    fig_annual = px.line(filtered_df, x='Year', y='Average annual working hours per worker', color='Entity',
                         title='Average Annual Working Hours per Worker')

    # Forecast for average annual working hours per worker
    for entity in selected_entities:
        entity_df = filtered_df[filtered_df['Entity'] == entity]
        forecast_annual = forecast_xgboost(entity_df, 'Average annual working hours per worker')
        forecast_years = list(range(entity_df['Year'].max() + 1, entity_df['Year'].max() + 1 + len(forecast_annual)))
        fig_annual.add_scatter(x=forecast_years, y=forecast_annual, mode='lines', name=f'{entity} Forecast')

    # Calculate average daily working hours per worker
    filtered_df['Average daily working hours per worker'] = filtered_df[
                                                                'Average annual working hours per worker'] / 365.24

    # Plot for average daily working hours per worker
    fig_daily = px.line(filtered_df, x='Year', y='Average daily working hours per worker', color='Entity',
                        title='Average Daily Working Hours per Worker')

    # Forecast for average daily working hours per worker
    for entity in selected_entities:
        entity_df = filtered_df[filtered_df['Entity'] == entity]
        forecast_daily = forecast_xgboost(entity_df, 'Average daily working hours per worker')
        forecast_years = list(range(entity_df['Year'].max() + 1, entity_df['Year'].max() + 1 + len(forecast_daily)))
        fig_daily.add_scatter(x=forecast_years, y=forecast_daily, mode='lines', name=f'{entity} Forecast')

    return fig_annual, fig_daily

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)