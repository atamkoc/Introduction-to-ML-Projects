from typing import Any
from activations import *
from utility import get_accuracy, get_predictions


class Model:
    def __init__(self, layer_dims: list = None, learning_rate: float = 0.002, num_epoch: int = 100,
                 batch_size: int = 16,
                 layer_activation: str = "relu", last_layer_act_func: str = "softmax", decay_rate: float = 0.998):
        """
        Initialize MultiLayerNeuralNetwork Model

        Args:
            layer_dims (list): An int list showing nodes for each layer, there should be at least 2 elements to create a model
            learning_rate (float): Learning rate of the model
            num_epoch (int): Number of epochs
            batch_size (int): Number of the batches
            layer_activation (str): Activation function that is going to be used for each hiden layer.
            last_layer_act_func (str): Activation function that is going to be used for the last layer.
            decay_rate (float): The values that is used for smoothing the training
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.layer_activation = layer_activation
        self.last_layer_act_func = last_layer_act_func
        self.decay_rate = decay_rate
        self.params = {}
        self.num_batch = 0
        self.current_batch = 0
        self.avg_acc = 0
        self.avg_loss = 0
        self.init_params()

    def init_params(self):
        """
        Sets the initial weights and biases for the model

        Args:
        Returns:
        """
        np.random.seed(42)

        for i in range(1, len(self.layer_dims)):
            self.params['W' + str(i)] = np.random.randn(self.layer_dims[i], self.layer_dims[i - 1]) * 0.001
            self.params['b' + str(i)] = np.zeros((self.layer_dims[i], 1))

    def forward_activation(self, Z: np.array, activation: str) -> Any:
        """
        A helper function to forward the activations depending on the model

        Args:
            Z (np.array): The data
            activation (str): The activation that is being used
        Returns:
            Any: Depends on the activation function that is being used
        """
        if activation == "sigmoid":
            return sigmoid(Z)
        elif activation == "relu":
            return relu(Z)
        elif activation == "softmax":
            return softmax(Z)
        elif activation == "tanh":
            return tanh(Z)

    def backward_activation(self, Z: np.array) -> Any:
        """
        A helper function to forward the activations depending on the model

        Args:
            Z (np.array): The data
        Returns:
            Any: Depends on the activation function that is being used
        """
        if self.layer_activation == "sigmoid":
            return derivative_sigmoid(Z)
        elif self.layer_activation == "relu":
            return derivative_relu(Z)
        elif self.layer_activation == "tanh":
            return derivative_tanh(Z)

    def forward(self, X: np.array) -> tuple[Any, Any]:
        """
        Forward activation function
        Makes the corresponding calculations and returns the results

        Args:
            X (np.array): The data
        Returns:
            Tuple[dict, dict]: Returns the tuple of layer_result and activation_result
        """
        layer_result = {}
        activation_result = {"A0": X}
        len_of_layer = len(self.layer_dims)
        for i in range(1, len_of_layer - 1):
            layer_result["Z" + str(i)] = np.dot(self.params['W' + str(i)], activation_result['A' + str(i - 1)]) + \
                                         self.params['b' + str(i)]
            activation_result["A" + str(i)] = self.forward_activation(layer_result["Z" + str(i)], self.layer_activation)

        layer_result["Z" + str(len_of_layer - 1)] = np.dot(self.params['W' + str(len_of_layer - 1)],
                                                           activation_result["A" + str(len_of_layer - 2)]) + \
                                                    self.params['b' + str(len_of_layer - 1)]
        activation_result["A" + str(len_of_layer - 1)] = self.forward_activation(
            layer_result["Z" + str(len_of_layer - 1)], self.last_layer_act_func)

        return layer_result, activation_result

    def backward(self, layer_result: dict, activation_result: dict, Y: np.array) -> dict:
        """
        Backward activation function
        Makes the corresponding calculations and returns the results

        Args:
            layer_result (dict): The layer calculations from forward activation
            activation_result (dict): The activation calculations from forward activation
            Y (np.array): The data batch from images corresponds to their classes
        Returns:
            derivate_results (dict): Returns the result of derivations
        """
        derivate_results = {}
        one_hot_Y = self.one_hot_encoding(Y)

        derivate_results["dZ" + str(len(self.layer_dims) - 1)] = activation_result[
                                                                     "A" + str(len(self.layer_dims) - 1)] - one_hot_Y
        derivate_results["dW" + str(len(self.layer_dims) - 1)] = (1 / self.batch_size) * np.dot(
            derivate_results["dZ" + str(len(self.layer_dims) - 1)],
            activation_result["A" + str(len(self.layer_dims) - 2)].T)
        derivate_results["db" + str(len(self.layer_dims) - 1)] = (1 / self.batch_size) * np.sum(
            derivate_results["dZ" + str(len(self.layer_dims) - 1)])

        for i in range(len(self.layer_dims) - 1, 1, -1):
            derivate_results["dZ" + str(i - 1)] = np.dot(self.params["W" + str(i)].T,
                                                         derivate_results["dZ" + str(i)]) * self.backward_activation(
                layer_result["Z" + str(i - 1)])
            derivate_results["dW" + str(i - 1)] = (1 / self.batch_size) * np.dot(derivate_results["dZ" + str(i - 1)],
                                                                                 activation_result["A" + str(i - 2)].T)
            derivate_results["db" + str(i - 1)] = (1 / self.batch_size) * np.sum(derivate_results["dZ" + str(i - 1)])

        return derivate_results

    def optimize(self, derivative_results: dict) -> None:
        """
        Depending on the backward activation results
        Optimizes and updates the weights and biases to have a better model
        Uses the decay rate for smoothing the result

        Args:
            derivative_results (dict): The derivative calculations from backward activation
        Returns:
        """
        for i in range(len(self.layer_dims) - 1):
            self.params["W" + str(i + 1)] -= (
                                                     self.learning_rate * derivative_results[
                                                 "dW" + str(i + 1)]) * self.decay_rate
            self.params["b" + str(i + 1)] -= (
                                                     self.learning_rate * derivative_results[
                                                 "db" + str(i + 1)]) * self.decay_rate

    def train(self, train_X: np.array, train_Y: np.array, valid_X: np.array, valid_Y: np.array) -> tuple[list, list]:
        """
        Train function for the model
        Uses training and validation data set while training
        Measures the accuracy and loss from validation data

        Args:
            train_X (np.array): The training data from the images
            train_Y (np.array): The training classes from the images
            valid_X (np.array): The validation data from the images
            valid_Y (np.array): The validation classes from the images
        Returns:
            tuple[list, list]: return accracy_list and loss_list from epochs to draw a graph later
        """
        loss_list = []
        accuracy_list = []
        for i in range(self.num_epoch):
            total_loss = 0
            self.num_batch = train_X.shape[0] // self.batch_size
            self.current_batch = 0
            while self.current_batch < self.num_batch:
                current_batch_X, current_batch_Y = self.get_next_batch(train_X, train_Y)

                layer_result, activation_result = self.forward(current_batch_X)
                derivate_results = self.backward(layer_result, activation_result, current_batch_Y)
                self.optimize(derivate_results)
                total_loss += self.compute_loss(current_batch_Y,
                                                activation_result["A" + str(len(activation_result) - 1)])

            _, valid_activation_result = self.forward(valid_X.T)
            predictions = get_predictions(valid_activation_result["A" + str(len(self.layer_dims) - 1)])
            acc = get_accuracy(predictions, valid_Y)
            accuracy_list.append(acc)
            loss_list.append(total_loss)

            print(f"Epoch {i + 1}/{self.num_epoch}\tAccuracy: {acc}\tTotal Loss: {total_loss}")
        self.avg_acc = np.average(accuracy_list)
        self.avg_loss = np.average(loss_list)
        return accuracy_list, loss_list

    def get_next_batch(self, train_X: np.array, train_Y: np.array) -> tuple[np.array, np.array]:
        """
        When it is called depending on the current batch and the batch size returns the next corresponding batch

        Args:
        Returns:
            tuple[np.array, np.array]: return accracy_list and loss_list from epochs to draw a graph later
        """
        begin = self.current_batch * self.batch_size
        end = begin + self.batch_size
        self.current_batch += 1

        return train_X.T[:, begin:end], train_Y[begin:end]

    def predict(self, test_X: np.array, test_Y: np.array) -> np.array:
        """
        Prediction functions to test the test data
        Measures the accuracy after predicting the values

        Args:
            test_X (np.array): The data from test images
            test_Y (np.array): The classes from test images
        Returns:
            predictions (np.array): Predictions
        """
        _, test_activation_result = self.forward(test_X.T)
        predictions = get_predictions(test_activation_result["A" + str(len(self.layer_dims) - 1)])
        acc = get_accuracy(predictions, test_Y)
        print("Accuracy:", acc)
        return predictions

    def one_hot_encoding(self, Y: np.array) -> np.array:
        """
        Normalization function

        Args:
            Y (np.array): The classes from images
        Returns:
            one_hot_Y (np.array): Normalized values
        """
        one_hot_Y = np.zeros((Y.size, self.layer_dims[-1]))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def compute_loss(self, train_Y: np.array, last_Activation: np.array) -> float:
        """
        Calculates negative log likelihood loss

        Args:
            train_Y (np.array): The classes from train images
            last_Activation (np.array): The results from last activation function
        Returns:
            (float): Loss value from the last activation
        """
        train_Y = self.one_hot_encoding(train_Y)
        L_sum = np.sum(np.multiply(train_Y, np.log(last_Activation)))
        m = train_Y.shape[1]
        return -(1. / m) * L_sum

    def display(self):
        """
        Displays the features of the current model

        Args:
        Returns:
        """
        print(f"Layer settings: {self.layer_dims}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Number of Epoch: {self.num_epoch}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Layer Activation: {self.layer_activation}")
        print(f"Last Layer Activation: {self.last_layer_act_func}")
        print(f"Decay rate: {self.decay_rate}")
