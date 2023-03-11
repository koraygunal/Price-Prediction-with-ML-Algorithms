from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
import pandas as pd

# load data
data = pd.read_csv("BTC-USD.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.drop(['Date'], axis=1)


# define the model
model = SVR()

# define the features and target
X = data.drop(columns=["RSI"])
y = data["RSI"]

# perform cross-validation
scores = cross_val_score(model, X, y, cv=5)

# print the mean and standard deviation of the scores
print("Mean: ", scores.mean())
print("Standard deviation: ", scores.std())
