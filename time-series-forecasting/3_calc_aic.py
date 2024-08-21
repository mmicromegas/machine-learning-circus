import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.datasets import co2
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Read the CSV file
df = pd.read_csv('DATA/bitcoin_monthly_close_updated.csv')
selected_entities = ['Bitcoin']
filtered_df = df[df['Currency'].isin(selected_entities)]

# Preprocess the DataFrame
for entity in selected_entities:
    entity_df = filtered_df[filtered_df['Currency'] == entity]
    entity_df = entity_df.set_index('Year_Month')
    entity_df.index = pd.to_datetime(entity_df.index.astype(str) + '-01')
    entity_df.index.name = None
    entity_df = entity_df.drop(columns=['Currency'])
    data = entity_df

    # Define a function to compute the AIC score for different ETS models
    def compute_aic(data, trend, seasonal, seasonal_periods):
        model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        fit = model.fit()
        return fit.aic

    # Iterate over different ETS models and compute their AIC scores
    ets_models = [
        ('add', 'add'),
        ('add', 'mul'),
        ('mul', 'add'),
        ('mul', 'mul')
    ]
    seasonal_periods = 12  # Adjust this based on your data

    aic_scores = {}
    for trend, seasonal in ets_models:
        try:
            aic = compute_aic(data, trend, seasonal, seasonal_periods)
            aic_scores[f'{trend}-{seasonal}'] = aic
        except Exception as e:
            print(f'Error for model {trend}-{seasonal}: {e}')

    # Print the AIC scores
    for model, aic in aic_scores.items():
        print(f'Model Holt-Winters (trend-seasonal): {model}, AIC score: {aic}')