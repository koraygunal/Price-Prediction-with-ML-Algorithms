import pandas as pd
import yfinance

# load historical stock price data
btc = yfinance.Ticker("BTC-USD")
df = btc.history(interval="1d",start="2022-12-17",end="2023-01-17")
df = df[['Close']]
lastprice = df.iloc[-1]
boundary = 500

# calculate the fibonacci retracement levels
high = 31176
low = 15606
levels = [0.236, 0.382, 0.5, 0.618, 0.786]

a=[]

for j in levels:
    valuee = high - (high - low) * j
    a.append(valuee)

print(a)


def signal():

    for i in range(0,5):
        if a[i] > lastprice > a[i] - boundary:
            print("SELL")
        elif a[i] + boundary > lastprice > a[i]:
            print("BUY")



signal()