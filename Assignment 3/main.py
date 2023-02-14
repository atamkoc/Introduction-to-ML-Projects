import pandas as pd
import utilities
from naive_bayes import NaiveBayes

CATEGORIES = {"sport": 1, "business": 2, "politics": 3, "entertainment": 4, "tech": 5}

# Reading CSV file
df = pd.read_csv("English Dataset.csv")
df = df.drop(["ArticleId"], axis=1)

# Setting int values for categories
cat = []
for index in range(len(df)):
    cat.append(CATEGORIES[df["Category"][index]])
df["Category"] = cat

Y = df["Category"].values
X = df["Text"].values
size = len(X)
X_train, X_test, y_train, y_test = X[:int(size*0.8)], X[int(size*0.8):], Y[:int(size*0.8)], Y[int(size*0.8):]

nb = NaiveBayes((2, 2), True)
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
utilities.calculate_accuracy(y_test, predictions)
