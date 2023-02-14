import KNN
import numpy as np
from utility import calculate_distance


class KNNClassifier(KNN.KNN):
    def predict(self, X, weighted=False):
        predicted_labels = [self._predict(x, weighted=weighted) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x, weighted=False):
        # Calculate the distance for every instance
        # Finding their indexes with the help from np.argsort()
        distances = [np.linalg.norm(x-x_train) for x_train in self.train_x]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = {}

        # In order to check the frequency, each appearance of the label increases its value 1
        # If the label is new, it is added to dictionary with value 1
        for i in k_indices:
            if self.train_y[i] in k_nearest_labels.keys():
                k_nearest_labels[self.train_y[i]] += 1
            else:
                k_nearest_labels[self.train_y[i]] = 1

        # If the KNN is weighted, then the result is distance dependent
        # Every label in dictionary is multiplied with 1/(its distance)
        # If the label is closer, then 1/distance is bigger so the new value is also bigger (comparably)
        if weighted:
            for i in k_indices:
                k_nearest_labels[self.train_y[i]] *= 1/distances[i]

        # Comparison of the existing labels in dictionary and finding the most frequent one regarding of its value
        most_frequent = None
        for key in k_nearest_labels.keys():
            if not most_frequent:
                most_frequent = key
            else:
                if k_nearest_labels[most_frequent] < k_nearest_labels[key]:
                    most_frequent = key
        return most_frequent
