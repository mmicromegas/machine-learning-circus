import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

# Read the CSV file
df = pd.read_csv('DATA/annual-working-hours-per-worker.csv')

# Ensure the Date column is in datetime format and set it as the index
df['Date'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Date', inplace=True)

# Check for and remove any duplicate labels in the date index
df = df[~df.index.duplicated(keep='first')]

# Check the frequency of the date index and set it explicitly if not inferred
if df.index.freq is None:
    df = df.asfreq('Y')

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Average Annual and Daily Working Hours per Worker"),
    dcc.Dropdown(
        id='entity-dropdown',
        options=[{'label': entity, 'value': entity} for entity in df['Entity'].unique()],
        value=['Slovakia'],
        multi=True
    ),
    dcc.Graph(id='annual-working-hours-graph'),
    dcc.Graph(id='daily-working-hours-graph'),
    dcc.Graph(id='trend-graph'),
    dcc.Graph(id='seasonal-graph'),
    dcc.Graph(id='residual-graph')
])

# Callback to update the graphs based on selected entities
@app.callback(
    [Output('annual-working-hours-graph', 'figure'),
     Output('daily-working-hours-graph', 'figure'),
     Output('trend-graph', 'figure'),
     Output('seasonal-graph', 'figure'),
     Output('residual-graph', 'figure')],
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

    # STL Decomposition function
    def stl_decomposition(df, column):
        data = df[column]
        print(data.head())
        print(data.shape)

        stl = STL(data, seasonal=13)

        result = stl.fit()
        return result

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

    # STL Decomposition plots
    stl_figures = {'trend': [], 'seasonal': [], 'residual': []}
    for entity in selected_entities:
        entity_df = filtered_df[filtered_df['Entity'] == entity]
        stl_result = stl_decomposition(entity_df, 'Average annual working hours per worker')

        stl_figures['trend'].append(px.line(x=entity_df.index, y=stl_result.trend, title=f'Trend for {entity}'))
        stl_figures['seasonal'].append(px.line(x=entity_df.index, y=stl_result.seasonal, title=f'Seasonal for {entity}'))
        stl_figures['residual'].append(px.line(x=entity_df.index, y=stl_result.resid, title=f'Residual for {entity}'))

    return fig_annual, fig_daily, stl_figures['trend'][0], stl_figures['seasonal'][0], stl_figures['residual'][0]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)