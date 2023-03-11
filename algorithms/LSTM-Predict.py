# Import the necessary libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Load the historical data on the price of Bitcoin
df = pd.read_csv("file.csv", parse_dates=['Date'], infer_datetime_format=True)
df = df[["Date", "Close"]]
df = df.iloc[::-1]
df = df.rename({"Date": "ds", "Close": "y"}, axis=1)
df["ds"] = pd.to_datetime(df["ds"])
df["y"] = df["y"]
# Extract the close price and convert it to a numpy array
df["y"] = df["y"].apply(lambda x: float(x.split()[0].replace(",", ".")))
close_prices = df["y"].values


# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.80  )
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Create the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Reshape the data for the LSTM model
X_train = np.reshape(train_data, (train_data.shape[0], 1, 1))
X_test = np.reshape(test_data, (test_data.shape[0], 1, 1))

# Train the model
model.fit(X_train, train_data, epochs=100, batch_size=1, verbose=2)

# Use the model to make predictions on the test data
predictions = model.predict(X_test)

# Reverse the scaling on the predictions
predictions = scaler.inverse_transform(predictions)
print(predictions)

# Print the predictions

pred = []
deneme = str(predictions)
pred.append(deneme)


kl = pd.DataFrame(predictions)
kl.to_csv("file_name19.csv",index=False)


