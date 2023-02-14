import random
import math


def train_test_split(x, y, split_ratio):
    split_index = math.floor(split_ratio * x.shape[0])
    train_x = x[:split_index]
    train_y = y[:split_index]
    test_x = x[split_index:]
    test_y = y[split_index:]

    return train_x, test_x, train_y, test_y


def calculate_distance(vector_1, vector_2):
    distance = 0
    for i in range(len(vector_1)):
        distance = distance + (vector_1[i] - vector_2[i]) ** 2

    return distance ** 0.5


def calculate_accuracy(predictions, test_y):
    true_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == test_y[i]:
            true_predictions += 1

    return (true_predictions / len(predictions)) * 100


def normalization(data_set):
    for key in data_set.keys():
        normalized_data = list()
        min_val = min(data_set[key])
        max_val = max(data_set[key])
        min_max_diff = max_val-min_val
        for row in range(len(data_set[key])):
            temp = abs((data_set[key][row] - min_val) / min_max_diff)
            normalized_data.append(temp)
            print(f"{row} for key {key}")
        data_set[key] = normalized_data
    return data_set


def shuffle(array):
    last_index = len(array) - 1
    while last_index > 0:
        rand_index = random.randint(0, last_index)
        array[last_index], array[rand_index] = array[rand_index], array[last_index]
        last_index = last_index - 1

    return array


def mean_absolute_error(predicted_values, test_y):
    d = 0
    for i in range(len(predicted_values)):
        d += abs(predicted_values[i] - test_y[i])
    return d / len(predicted_values)


def recall(predictions, test_y):
    TP = {}
    FN = {}

    for i in range(len(predictions)):
        if test_y[i] not in TP:
            TP[test_y[i]] = 0
            FN[test_y[i]] = 0
        if test_y[i] == predictions[i]:
            TP[test_y[i]] += 1
        else:
            FN[test_y[i]] += 1

    result = 0
    for key in TP.keys():
        result = TP[key] / (TP[key] + FN[key])
    return result * 100


def precision(predictions, test_y):
    TP = {}
    FP = {}

    for i in range(len(test_y)):
        if predictions[i] not in TP:
            TP[predictions[i]] = 0
            FP[predictions[i]] = 0
        if test_y[i] == predictions[i]:
            TP[predictions[i]] += 1
        else:
            FP[predictions[i]] += 1

    result = 0
    for key in TP.keys():
        result = TP[key] / (TP[key] + FP[key])
    return result * 100


def display_result(k, regress_val=0, accuracy=0, recall=0, precision=0, normalization=False, weighted=False,
                   regression=False):
    print(f"For k = {k}, Normalization = {normalization} and Weighted = {weighted}", end="\n")
    if regression:
        print(f"Mean Absolute Error = {regress_val}")
    else:
        print(f"Accuracy = {accuracy}")
        print(f"Recall = {recall}")
        print(f"Precision = {precision}")
        print("*" * 20)
