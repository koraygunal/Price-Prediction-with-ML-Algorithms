import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

kl = pd.read_csv("denemebtc.csv")
df = pd.read_csv("denemebtc.csv")

df = pd.read_csv("btc.csv", parse_dates=['Date'], infer_datetime_format=True)
df = df[["Date", "Price"]]
df = df.rename({"Date": "ds", "Price": "y"}, axis="columns")
df["ds"] = pd.to_datetime(df["ds"])
df["y"] = df["y"].apply(lambda x: float(x.split()[0].replace(",", "")))

# Transfer column 'x' from df1 to df2
kl = pd.concat([kl, df['y']], axis=1)

# Save df2 as CSV
kl.to_csv('file.csv')

print(df.head())
print(kl.head())