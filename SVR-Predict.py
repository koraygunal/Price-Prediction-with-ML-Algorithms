import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import time

data = pd.read_csv("BTC-USD.csv")

# UNIX hale çevirmek
data['Date'] = data['Date'].apply(lambda x: time.mktime(pd.to_datetime(x).timetuple()))


data = data[["Price","Volume","RSI"]]
X = data.drop(["Price"], axis=1)
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=101)

model = SVR()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = abs(predictions - y_test).mean()
print("Mean Absolute Error: ", mae)
print(predictions)


