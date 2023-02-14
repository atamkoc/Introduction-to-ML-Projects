import KNNClassifier
import KNNRegressor
import utility


class KFoldCV:
    def __init__(self, is_regression: bool, data_set, num_folds: int):
        self.is_regression = is_regression
        self.data_set = data_set
        self.num_folds = num_folds

    def cross_validation_split(self):
        dataSplit = list()
        foldSize = int(len(self.data_set) / self.num_folds)
        for index in range(self.num_folds):
            fold = self.data_set[index*foldSize:(index+1)*foldSize]
            dataSplit.append(fold)
        return dataSplit

    def evaluate(self, weighted=False):
        if self.is_regression:
            knn = KNNClassifier.KNNClassifier(5)
        else:
            knn = KNNRegressor.KNNRegressor(5)
        folds = self.cross_validation_split()
        scores = list()
        for num in range(self.num_folds):
            trainSet = list(folds)
            trainSet.remove(folds[num])
            trainSet = sum(trainSet, [])
            testSet = list(folds[num])

            train_labels = [row[-1] for row in trainSet]
            trainSet = [train[:-1] for train in trainSet]
            knn.fit(trainSet, train_labels)

            actual = [row[-1] for row in testSet]
            testSet = [test[:-1] for test in testSet]

            result = None
            if self.is_regression:
                result = knn.regression(testSet, weighted=weighted)
            else:
                result = knn.predict(testSet, weighted=weighted)

            accuracy = utility.calculate_accuracy(actual, result)
            scores.append(accuracy)
