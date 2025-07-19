import sys

from statsmodels.datasets import co2
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import pandas as pd

from statsmodels.tsa.seasonal import STL

# STL Decomposition function
def stl_decomposition(data):
    stl = STL(data)
    result = stl.fit()
    return result

# Read the CSV file
df = pd.read_csv('DATA/annual-working-hours-per-worker.csv')
selected_entities = ['Slovakia']
filtered_df = df[df['Entity'].isin(selected_entities)]



# STL Decomposition plots
stl_figures = {'trend': [], 'seasonal': [], 'residual': []}
for entity in selected_entities:
    entity_df = filtered_df[filtered_df['Entity'] == entity]

    print(entity_df.head())

    # in entity_df, push Year to index and keep only the column Average annual working hours per worker but keep its name
    entity_df = entity_df.set_index('Year')

    # to the index, add month and day of january the first
    entity_df.index = pd.to_datetime(entity_df.index.astype(str) + '-01-01')

    # remove name of the index
    entity_df.index.name = None

    # delete columns Entity and Code
    entity_df = entity_df.drop(columns=['Entity', 'Code'])

    print(entity_df.head())

    data = entity_df



    print(data)
    print(data.shape)

    stl = STL(data,period=24)
    result = stl.fit()

    result.plot()
    plt.show()


    print(entity_df.index)
    print("-----------")
    print(result.trend)

    #plt.plot(entity_df.index,result.trend)
    #plt.plot(entity_df.index,result.seasonal)
    #plt.plot(entity_df.index,result.resid)
    #plt.plot(entity_df.index,filtered_df[filtered_df['Entity'] == entity]['Average annual working hours per worker'])

    # plot trend + seasonal + residual with dashed line
    #plt.plot(entity_df.index, result.trend + result.seasonal + result.resid, linestyle='--')


    # plot legend
    #plt.legend(['Trend', 'Seasonal', 'Residual', 'Original','Sum'])

    #plt.show()







