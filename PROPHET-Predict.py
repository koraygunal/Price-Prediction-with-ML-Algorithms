import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
import datetime

# load data
df = pd.read_csv("reversed.csv",parse_dates=['Date'], infer_datetime_format=True)
'''df = df[["Date", "Price"]]
df = df.rename({"Date": "ds", "Price": "y"}, axis="columns")
print(df.head())'''


df = df[["Date", "Price"]]
df = df.rename({"Date": "ds", "Price": "y"}, axis="columns")
df["ds"] = pd.to_datetime(df["ds"])
df["y"] = df["y"].apply(lambda x: float(x.split()[0].replace(",",".")))
print(df.head())
# define the model
model = Prophet()

# fit the model
model.fit(df)

# create future dataframe
future = model.make_future_dataframe(periods=10)

# predict
forecast = model.predict(future)

# plot the forecast
model.plot(forecast)
plt.show()
print(forecast)

forecast.to_csv('file_name1.csv',index=False)
