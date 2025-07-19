import matplotlib.pyplot as plt

import pandas as pd
import random
import numpy as np


# create a dataframe with random average transaction values for 12 months of the year
def create_dataframe():
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    random.seed(0)
    avg_transaction = [random.randint(100, 1000) for i in range(6)]
    df = pd.DataFrame({'Month': months, 'Total Transaction': avg_transaction})
    # fix the random seed to make the plot reproducible

    return df

# plot the dataframe
def plot_dataframe(df):
    plt.plot(df['Month'], df['Total Transaction'])
    # add x label called month, add y label called Total Transaction
    plt.xlabel('Month')
    plt.ylabel(r'Total Transactions (monthly value) $\Sigma_\tau$')

    # oplot mean and stdev
    plt.axhline(y=np.mean(df['Total Transaction']), color='r', linestyle='-')
    plt.axhline(y=np.mean(df['Total Transaction']) + np.std(df['Total Transaction']), color='g', linestyle='--')
    plt.axhline(y=np.mean(df['Total Transaction']) - np.std(df['Total Transaction']), color='g', linestyle='--')

    # add two new points to the dataframe for January and February and plot them
    df = df._append({'Month': 'Jul', 'Total Transaction': 1000}, ignore_index=True)
    df = df._append({'Month': 'Aug', 'Total Transaction': 2000}, ignore_index=True)
    plt.plot(df['Month'], df['Total Transaction'], 'ro')

    # insert sigma_c^tau to the upper right corner of the plot
    plt.text(6.4, 1900., r'$\Sigma^c_{\tau}$', fontsize=20, color='black')

    # insert sigma to the lower right corner of the plot
    plt.text(0.5, 860., r'$\sigma$', fontsize=20, color='black')

    # insert tau to the upper left corner of the plot
    plt.text(0.2, 400., r'$\overline{\tau}$', fontsize=20, color='black')

    # make the size of the plot bigger
    plt.gcf().set_size_inches(8, 6)

    plt.show()


plot_dataframe(create_dataframe())

