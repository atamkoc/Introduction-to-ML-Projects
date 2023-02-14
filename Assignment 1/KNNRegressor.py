import numpy as np
import pandas as pd
import KNN
from utility import calculate_distance


class KNNRegressor(KNN.KNN):
    def regression(self, X, weighted=False):
        regression = [self._regression(x, weighted) for x in X]
        return regression

    def _regression(self, x, weighted=False):
        # Calculate the distance for every instance
        # Finding their indexes with the help from np.argsort()
        # It also creates the array of the values from the indexes
        distances = [calculate_distance(x, x_train) for x_train in self.train_x]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_values = [self.train_y[i] for i in k_indices]
        result = 0

        # If the regression is weighted the value is multiplied with 1/distance
        # So if the instance is closer, it will have more affect on the result
        # Since as distance gets lower, 1/distance gets higher
        if weighted:
            weightSum = 0
            for index in range(len(k_nearest_values)):
                weight = 1 / distances[k_indices[index]]
                weightSum += weight
                result += k_nearest_values[index] * weight
            # Bunu bi kontrol et
            return result/weightSum
        # If the regression is not weighted, the values sum up and divided by K value to find the average
        else:
            for value in k_nearest_values:
                result += value
            return result / self.k
