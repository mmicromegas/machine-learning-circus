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
df = pd.read_csv('DATA/bitcoin_monthly_close_updated.csv')
selected_entities = ['Bitcoin']
filtered_df = df[df['Currency'].isin(selected_entities)]



# STL Decomposition plots
stl_figures = {'trend': [], 'seasonal': [], 'residual': []}
for entity in selected_entities:
    entity_df = filtered_df[filtered_df['Currency'] == entity]

    # in entity_df, push Year to index and keep only the column Average annual working hours per worker but keep its name
    entity_df = entity_df.set_index('Year_Month')

    # to the index, add month and day of january the first
    entity_df.index = pd.to_datetime(entity_df.index.astype(str) + '-01')

    # remove name of the index
    entity_df.index.name = None

    # delete columns Entity and Code
    entity_df = entity_df.drop(columns=['Currency'])

    data = entity_df

    stl = STL(data,period=9)
    result = stl.fit()

    # plot result of STL decomposition with new title Bitcoin Monthly Close
    result.plot()


    plt.show()


    # create new figure 2 with title Bitcoin Monthly Close
    #plt.figure(2)

    #plt.title('Bitcoin Monthly Close')
    #plt.plot(entity_df)

    # add x axis label to be Year
    #plt.xlabel('Year')

    # add y axis label to be Monthly Close (USD)
    #plt.ylabel('Monthly Close (USD)')


    #plt.show()







