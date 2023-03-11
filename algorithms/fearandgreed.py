import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pylab import rcParams
import warnings

warnings.filterwarnings('ignore')
'''def fear_greed():'''
rcParams['figure.figsize'] = 10, 5
r = requests.get('https://api.alternative.me/fng/?limit=0')
df = pd.DataFrame(r.json()['data'])
df = df.loc[0:365]
value = df["value"].iloc[0]
df.value = df.value.astype(int)
df.timestamp = pd.to_datetime(df.timestamp, unit='s')
df.set_index(df.timestamp, inplace=True)
df.rename(columns={'value': 'fear_greed'}, inplace=True)
df.drop(['timestamp', 'time_until_update'], axis=1, inplace=True)
df.fear_greed.plot(figsize=(20, 10))
plt.show()
df.to_csv("yeni.csv")
pl = pd.read_csv("yeni.csv", index_col=False)
pl = pl.iloc[::-1]

'''return value

print(fear_greed())'''

df1 = yf.download('BTC-USD', interval = '1d')[['Close']]
df1.rename(columns = {'Close':'close'}, inplace=True)
df1.index.name = 'timestamp'
df1['timestamp'] = df1.index
df1.reset_index(drop=True, inplace=True)
df1.timestamp = pd.to_datetime(df1.timestamp, unit='s').dt.tz_localize(None)
df1.set_index(df1.timestamp, inplace=True)
df1.drop(['timestamp'], axis=1, inplace=True)
data = df.merge(df1, on='timestamp')
data = data.sort_index()
data.tail()
data['close_tomorrow'] = data['close'].shift(-1)
data['returns'] = data['close_tomorrow'] / data['close'] - 1
data['change_btc'] = (data['returns'] + 1).cumprod()
data = data.dropna()
data.to_csv("onemli.csv")
print(data['fear_greed'].corr(data['close ']))
