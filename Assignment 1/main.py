import pandas as pd
import numpy as np

import KNNRegressor
import utility
import KNNClassifier

k_list = [1]

# importing our dataset as dataframe and getting rid of response id columns because
# it is irrelevant for our model
df = pd.read_csv("subset_16P.csv")
df = df.drop(["Response Id"], 1)
df = df.head(1000)
y = np.array(df["Personality"])
X = np.array(df.drop(["Personality"], 1))
X_train, X_test, y_train, y_test = utility.train_test_split(X, y, 0.8)

normalized_df = utility.normalization(df.drop(["Personality"], 1))
normalized_X = np.array(normalized_df)
X_train_norm, X_test_norm, y_train_norm, y_test_norm = utility.train_test_split(normalized_X, y, 0.8)

df2 = pd.read_csv("energy_efficiency_data.csv")
y2 = np.array(df2["Cooling_Load"])
X2 = np.array(df2.drop(["Cooling_Load"], 1))
X_train2, X_test2, y_train2, y_test2 = utility.train_test_split(X2, y2, 0.8)


for k in k_list:
    knn = KNNClassifier.KNNClassifier(k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = utility.calculate_accuracy(predictions, y_test)
    recall = utility.recall(predictions, y_test)
    precision = utility.precision(predictions, y_test)
    utility.display_result(k, accuracy=accuracy, recall=recall, precision=precision)

    predictions = knn.predict(X_test, weighted=True)
    accuracy = utility.calculate_accuracy(predictions, y_test)
    recall = utility.recall(predictions, y_test)
    precision = utility.precision(predictions, y_test)
    utility.display_result(k, accuracy=accuracy, recall=recall, precision=precision, weighted=True)

    knn.fit(X_train_norm, y_train_norm)
    predictions = knn.predict(X_test_norm)
    accuracy = utility.calculate_accuracy(predictions, y_test_norm)
    recall = utility.recall(predictions, y_test_norm)
    precision = utility.precision(predictions, y_test_norm)
    utility.display_result(k, accuracy=accuracy, recall=recall, precision=precision, normalization=True)

    predictions = knn.predict(X_test_norm, weighted=True)
    accuracy = utility.calculate_accuracy(predictions, y_test_norm)
    recall = utility.recall(predictions, y_test_norm)
    precision = utility.precision(predictions, y_test_norm)
    utility.display_result(k, accuracy=accuracy, recall=recall, precision=precision, weighted=True, normalization=True)

    regressor = KNNRegressor.KNNRegressor(k)
    regressor.fit(X_train2, y_train2)
    prediction_values = regressor.regression(X_test2)
    mae = utility.mean_absolute_error(prediction_values, y_test2)
    utility.display_result(k, regress_val=mae, regression=True)

    prediction_values = regressor.regression(X_test2, weighted=True)
    mae_weighted = utility.mean_absolute_error(prediction_values, y_test2)
    utility.display_result(k, regress_val=mae, regression=True, weighted=True)






