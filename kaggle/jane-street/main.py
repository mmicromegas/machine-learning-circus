import os
import pandas

# show content of folder DATA
print(os.listdir('DATA'))

# read part-0.parquet from DATA\train.parquet\partition_id=0 folder
df = pandas.read_parquet('DATA/train.parquet/partition_id=1/part-0.parquet')

print(df['feature_00'].unique())
print(df.columns)

print(df['date_id'].unique())

df_lags = pandas.read_parquet('DATA/lags.parquet/date_id=0/part-0.parquet')

print(df_lags.columns)

print(df_lags.head())


df_test = pandas.read_parquet('DATA/test.parquet/date_id=0/part-0.parquet')

print(df_test.columns)

print(df_test.head())