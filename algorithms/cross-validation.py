import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import time

# load historical stock price data
df = pd.read_csv("tupras.csv")
df = df.iloc[::-1]
df["Volume"] = df["Volume"].apply(lambda x: float(x.split()[0].replace("M", "")))
print(df["Volume"])

df = df[["Price","Volume","Rate"]]
# extract features and labels
X = df.drop(["Price"], axis=1)
y = df["Price"]

# initialize the model
model = RandomForestRegressor()

# perform k-fold cross-validation
k = 5 # number of folds
scores = cross_val_score(model, X, y, cv=k)

# print the mean and standard deviation of the scores
print("Mean score: {:.2f}".format(scores.mean()))
print("Standard deviation: {:.2f}".format(scores.std()))
