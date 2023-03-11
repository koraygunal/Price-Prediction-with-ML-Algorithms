from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# load the data into a pandas dataframe
df = pd.read_csv('BTC-USD.csv')

# extract the features and labels
X = df[['Price', 'RSI', 'Volume']]
y = df['Price']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.14)
# initialize the base model
base_model = DecisionTreeRegressor()

# initialize the bagging ensemble
ensemble = BaggingRegressor(base_estimator=base_model, n_estimators=10)

# fit the ensemble to the data
ensemble.fit(X, y)

# make predictions
y_pred = ensemble.predict(X_test)
print(y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)
