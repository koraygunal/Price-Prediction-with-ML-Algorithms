import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time

# load historical bitcoin price data
data = pd.read_csv("reversed.csv")
data = data.drop(["Volume"],axis=1)
data = data.drop(["Rate"],axis=1)

data['Date'] = data['Date'].apply(lambda x: time.mktime(pd.to_datetime(x,infer_datetime_format=True).timetuple()))

# extract features and labels
X = data.drop(["Price"], axis=1)
y = data["Price"]

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# initialize the model
model = RandomForestRegressor()

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the test data
predictions = model.predict(X_test)

# evaluate the model's performance
mae = abs(predictions - y_test).mean()
print("Mean Absolute Error: ", mae)
print(predictions)