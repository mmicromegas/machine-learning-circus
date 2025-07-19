import pandas as pd
import os
from matplotlib import pyplot as plt

# Create path with os.join to DATA/tipos folder
path = os.path.join('DATA', 'tipos')

# Read the keno10.csv file from the path
df = pd.read_csv(os.path.join(path, 'keno10.csv'), sep=';')

print(df.head())

# Combine all numbers from C_1 to C_20 into one Series
all_numbers = df.iloc[:, 3:23].values.flatten()

# Plot histogram of all numbers
#plt.hist(all_numbers, bins=20)
#lt.xlabel('Number')
#plt.ylabel('Frequency')
#plt.title('Distribution of Numbers C_1 to C_20')
#plt.show()

# cast DATUM to datetime
df['DATUM'] = pd.to_datetime(df['DATUM'],format='%d.%m.%Y')

# plot the frequency of numbers C_1 to C_20 for each year in DATUM field
#for year in df['DATUM'].dt.year.unique():
#    year_df = df[df['DATUM'].dt.year == year]
#    all_numbers = year_df.iloc[:, 3:23].values.flatten()
#    plt.hist(all_numbers, bins=20)
#    plt.xlabel('Number')
#    plt.ylabel('Frequency')
#    plt.title(f'Distribution of Numbers C_1 to C_20 for year {year}')
#    plt.show()


# plot the frequency of numbers C_1 to C_20 for each year in DATUM field in
