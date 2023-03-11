import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('BTC-USD.csv')


X = df[['Open', 'High', 'Volume']]
y = df['RSI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.46)


regr = SVR()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)
