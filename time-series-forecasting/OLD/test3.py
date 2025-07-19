import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Read the CSV file
df = pd.read_csv('../DATA/monthly_close_5_years.csv')

# Function to fit ETS models and compute AIC scores
def compute_aic_scores(df, column):
    models = {
        'AAN': ExponentialSmoothing(df[column], trend='add', seasonal=None),
        'AAdN': ExponentialSmoothing(df[column], trend='add', damped_trend=True, seasonal=None),
        'ANA': ExponentialSmoothing(df[column], trend=None, seasonal='add', seasonal_periods=12),
        'ANM': ExponentialSmoothing(df[column], trend=None, seasonal='mul', seasonal_periods=12),
        'AAM': ExponentialSmoothing(df[column], trend='add', seasonal='mul', seasonal_periods=12),
        'AAdA': ExponentialSmoothing(df[column], trend='add', damped_trend=True, seasonal='add', seasonal_periods=12),
        'AAdM': ExponentialSmoothing(df[column], trend='add', damped_trend=True, seasonal='mul', seasonal_periods=12),
        'MAN': ExponentialSmoothing(df[column], trend='mul', seasonal=None),
        'MAdN': ExponentialSmoothing(df[column], trend='mul', damped_trend=True, seasonal=None),
        'MNA': ExponentialSmoothing(df[column], trend=None, seasonal='add', seasonal_periods=12),
        'MNM': ExponentialSmoothing(df[column], trend=None, seasonal='mul', seasonal_periods=12),
        'MAM': ExponentialSmoothing(df[column], trend='mul', seasonal='mul', seasonal_periods=12),
        'MAdA': ExponentialSmoothing(df[column], trend='mul', damped_trend=True, seasonal='add', seasonal_periods=12),
        'MAdM': ExponentialSmoothing(df[column], trend='mul', damped_trend=True, seasonal='mul', seasonal_periods=12)
    }

    aic_scores = {}
    for model_name, model in models.items():
        fit = model.fit()
        aic_scores[model_name] = fit.aic

    return aic_scores

# Compute AIC scores for the 'Close_Price' column
aic_scores = compute_aic_scores(df, 'Close_Price')

# Determine the model with the lowest AIC score
best_model = min(aic_scores, key=aic_scores.get)

# Print the best model with explicit reasons
error, trend, seasonal = best_model[0], best_model[1:], best_model[2:]
trend = 'No Trend' if trend == 'N' else 'Additive' if trend == 'A' else 'Additive damped' if trend == 'Ad' else 'Multiplicative'
seasonal = 'No seasonality' if seasonal == 'N' else 'Additive' if seasonal == 'A' else 'Multiplicative'

print(f"Best model: {best_model} with AIC score: {aic_scores[best_model]}")
print(f"Error: {'Additive' if error == 'A' else 'Multiplicative'}")
print(f"Trend: {trend}")
print(f"Seasonal: {seasonal}")