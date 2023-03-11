import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("reversed.csv", parse_dates=['Date'], infer_datetime_format=True)
df = df[["Date", "Price"]]
df = df.rename({"Date": "ds", "Price": "y"}, axis=1)
df["ds"] = pd.to_datetime(df["ds"])
df["y"] = df["y"].apply(lambda x: float(x.split()[0].replace(",", ".")))

# load historical bitcoin price data

# calculate moving average
window_size = 14
moving_average = df['y'].rolling(window=window_size).mean()

# add the moving average column to the dataframe
df['moving_average'] = moving_average

# make predictions
predictions = df.iloc[-window_size:]['moving_average']

# print the predictions
print(predictions)



df[['y','moving_average']].plot(subplots=False,figsize=(12,5))

plt.show()




