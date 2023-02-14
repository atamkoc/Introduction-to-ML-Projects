import numpy as np


def relu(Z: np.ndarray) -> np.ndarray:
    """
    Calculates the ReLU for forward activation

    Args:
        Z (np.ndarray): input, shape (N,D)
    Returns:
        (np.ndarray): output, shape (N,D)
    """
    return np.maximum(0, Z)


def derivative_relu(Z: np.ndarray) -> np.ndarray:
    """
    Calculates the ReLU for backward activation

    Args:
        Z (np.ndarray): input, shape (N,D)
    Returns:
        (np.ndarray): output, shape (N,D)
    """
    return np.greater(Z, 0).astype(int)


def tanh(Z: np.ndarray) -> np.ndarray:
    """
    Calculates the Tanh value for forward activation

    Args:
        Z (np.ndarray): input, shape (N,D)
    Returns:
        (np.ndarray): output, shape (N,D)
    """
    z = np.tanh(Z)
    z = np.asarray(z)
    z[z < 1e-20] = 1e-20

    return z


def derivative_tanh(Z: np.ndarray) -> np.ndarray:
    """
    Calculates the Tanh value for backward activation

    Args:
        Z (np.ndarray): input, shape (N,D)
    Returns:
        (np.ndarray): output, shape (N,D)
    """
    Z = tanh(Z)
    Z = (1 - np.power(Z, 2))
    return Z


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Calculates the Sigmoid value for forward activation

    Args:
        Z (np.ndarray): input, shape (N,D)
    Returns:
        (np.ndarray): output, shape (N,D)
    """
    return 1 / (1 + np.exp(-Z))


def derivative_sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Calculates the Sigmoid value for backward activation

    Args:
        Z (np.ndarray): input, shape (N,D)
    Returns:
        (np.ndarray): output, shape (N,D)
    """
    sigmo = sigmoid(Z)
    return sigmo * (1 - sigmo)


def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Calculates the Softmax value for last layer activation

    Args:
        Z (np.ndarray): input, shape (N,D)
    Returns:
        (np.ndarray): output, shape (N,D)
    """
    Z = np.exp(Z - np.max(Z))
    return Z / np.sum(Z, axis=0)
