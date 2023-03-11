import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# load historical bitcoin price data
df = pd.read_csv("reversed.csv",parse_dates=['Date'], infer_datetime_format=True)

####
df = df[["Date", "Price"]]
df = df.rename({"Date": "ds", "Price": "y"}, axis="columns")
df["ds"] = pd.to_datetime(df["ds"])
df["y"] = df["y"].apply(lambda x: float(x.split()[0].replace(",",".")))

##

# fit an ARIMA model to the data
model = ARIMA(df["y"], order=(2, 1, 2))
model_fit = model.fit()

# make predictions
predictions = model_fit.predict(start=len(df), end=len(df)+14, typ='levels')

# print the predictions
print(predictions)
