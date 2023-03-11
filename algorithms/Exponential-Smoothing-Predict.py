import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# load historical bitcoin price data
data = pd.read_csv("BTC-USD.csv")

# split data into training and test sets
train_data = data[:int(len(data)*0.33)]
test_data = data[int(len(data)*0.33):]

# fit an Exponential Smoothing model to the training data
model = ExponentialSmoothing(train_data["Price"])
model_fit = model.fit()

# make predictions on the test data
predictions = model_fit.predict(start=len(train_data), end=len(data)-1)

# evaluate the model's performance
mae = abs(predictions - test_data["Price"]).mean()
print("Mean Absolute Error: ", mae)
print(predictions)
