import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.elm import ELMRegressor

# load stock data into a pandas dataframe
df = pd.read_csv('BTC-USD.csv')

# extract features and labels
X = df[['Open', 'Close', 'Volume']]
y = df['Price']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train an ELM model on the training data
elm = ELMRegressor(n_hidden=100)
elm.fit(X_train, y_train)

# make predictions on the test data
y_pred = elm.predict(X_test)

# evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
print(mae)