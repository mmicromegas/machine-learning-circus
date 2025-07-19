# https://medium.com/@sachin.jain3001/exploring-exponential-smoothing-techniques-for-time-series-forecasting-in-python-e109a010d22d

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing import set_font_settings_for_testing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt

# load the airpassenger dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# from monthly_close_5_years.csv, read all data for Bitcoin
df = pd.read_csv('DATA/monthly_close_5_years.csv')
df = df[df['Cryptocurrency'] == 'Bitcoin']
# push Date to index
df = df.set_index('Date')

data = df

# remove columns Cryptocurrency
data = data.drop(columns=['Cryptocurrency'])

# read bitcoin_monthly_close.csv
data = pd.read_csv('DATA/bitcoin_monthly_close.csv')

# push Year_Month to index
data = data.set_index('Year_Month')

print(data.head())

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

ses_model = SimpleExpSmoothing(train).fit()
ses_forecast = ses_model.forecast(steps=len(test))

holt_model = Holt(train).fit()
holt_forecast = holt_model.forecast(steps=len(test))

hw_model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12).fit()
hw_forecast = hw_model.forecast(steps=len(test))

# plot the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, ses_forecast, label='Simple Exponential Smoothing')
plt.plot(test.index, holt_forecast, label='Holt')
plt.plot(test.index, hw_forecast, label='Holt-Winters')


plt.legend()
plt.show()