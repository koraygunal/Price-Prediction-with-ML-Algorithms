import pandas as pd
import numpy as np
df = pd.read_csv("BTC-USD.csv")

def rsi(df, periods=14, ema=True):
    close_delta = df['Close'].diff()

    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema == True:
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    else:        # Use simple moving average
        ma_up = up.rolling(window=periods, adjust=False).mean()
        ma_down = down.rolling(window=periods, adjust=False).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    rsilastvalue= rsi[len(rsi) - 1]
    print(rsilastvalue)


    return rsilastvalue



def signal():
    if rsi_signal < 30 :
        print("BUY")
    elif rsi_signal > 80:
        print("SELL")
    else:
        print("STABILE")


rsi_signal = rsi(df)
signal()
