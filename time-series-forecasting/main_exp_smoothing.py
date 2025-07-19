import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Read the CSV file
df = pd.read_csv('DATA/annual-working-hours-per-worker.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
#app.layout = html.Div([
#    html.H1("Average Annual and Daily Working Hours per Worker"),
#    dcc.Dropdown(
#        id='entity-dropdown',
#        options=[{'label': entity, 'value': entity} for entity in df['Entity'].unique()],
#        value=[df['Entity'].unique()[0]],
#        multi=True
#    ),
#    dcc.Graph(id='annual-working-hours-graph'),
#    dcc.Graph(id='daily-working-hours-graph')
#])

# make the default selection in dropdown to be Slovakia
app.layout = html.Div([
     html.H1("Average Annual and Daily Working Hours per Worker"),
     dcc.Dropdown(
         id='entity-dropdown',
         options=[{'label': entity, 'value': entity} for entity in df['Entity'].unique()],
         value=['Slovakia'],
         multi=True
     ),
     dcc.Graph(id='annual-working-hours-graph'),
     dcc.Graph(id='daily-working-hours-graph')
])


# Callback to update the graphs based on selected entities
@app.callback(
    [Output('annual-working-hours-graph', 'figure'),
     Output('daily-working-hours-graph', 'figure')],
    [Input('entity-dropdown', 'value')]
)
def update_graphs(selected_entities):
    filtered_df = df[df['Entity'].isin(selected_entities)]

    # Forecasting function
    def forecast(df, column, periods=100):
        model = ExponentialSmoothing(df[column], trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(periods)
        return forecast

    # Plot for average annual working hours per worker
    fig_annual = px.line(filtered_df, x='Year', y='Average annual working hours per worker', color='Entity',
                         title='Average Annual Working Hours per Worker')

    # Forecast for average annual working hours per worker
    for entity in selected_entities:
        entity_df = filtered_df[filtered_df['Entity'] == entity]
        forecast_annual = forecast(entity_df, 'Average annual working hours per worker')
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
        forecast_daily = forecast(entity_df, 'Average daily working hours per worker')
        forecast_years = list(range(entity_df['Year'].max() + 1, entity_df['Year'].max() + 1 + len(forecast_daily)))
        fig_daily.add_scatter(x=forecast_years, y=forecast_daily, mode='lines', name=f'{entity} Forecast')

    return fig_annual, fig_daily


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)