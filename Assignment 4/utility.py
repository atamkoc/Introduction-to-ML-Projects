import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def read(data_set_folder: str = "Vegetable Images", mode: str = "train", model_name: str = None) -> Any:
    """
    Reads the data or given model from specified path
    If there is saved files, reads the data from those files
    This makes the process much faster
    If not, reads the files from the sources

    Args:
        data_set_folder (str): Folder name where all the files (test/train/validation) are in
        mode (str): Specifies which files it tries to read, can be test, train, validation and model
        model_name (str): In case of reading a saved model, specifies the model's file name without extension
    Returns:
        tuple[Any, Any]: Returns 2 np.arrays when it reads the image files
        Model: Returns a model, when it reads a saved model file
    """
    print(f"Trying to load {mode} files from previous saves")
    if model_name:
        try:
            model_params = load(mode=mode, model_name=model_name)
            return model_params
        except FileNotFoundError:
            print(f"There is not any file save for {model_name}")
    else:
        try:
            X, Y = load(mode=mode)
        except FileNotFoundError:
            print(f"Could not find {mode} save files")
            print("Reading the files from the source")
            # Sets the path properly depending on the mode and data set folder
            path = os.path.join(os.getcwd(), data_set_folder)
            path = os.path.join(path, mode)
            directories = [name for name in os.listdir(path)]

            X, Y = [], []
            for i in range(len(directories)):
                dir_path = os.path.join(path, directories[i])
                files = os.listdir(dir_path)
                for image in files:
                    im = Image.open(dir_path + "\\" + image)
                    im = im.resize((120, 120))
                    # Converts the images in gray scale
                    im = im.convert('L')
                    # PIL to numpy array
                    data = np.asarray(im)
                    data = data.flatten()
                    data = np.divide(data, 255)
                    X.append(data)
                    Y.append(i)

            # Mixes the data before sending it to the model
            X = np.array(X, dtype=float)
            Y = np.array(Y, dtype=int)
            save(X, path=mode + "_X")
            save(Y, path=mode + "_Y")

        return _randomize(X, Y)


def _randomize(X: np.array, Y: np.array) -> tuple[Any, Any]:
    """
    After reading the data from either source or saved files, it mixes the arrays for better training

    Args:
        X (np.array): Folder name where all the files (test/train/validation) are in
        Y (np.array): Specifies which files it tries to read, can be test, train, validation and model
    Returns:
        tuple[Any, Any]: Returns both mixed X and Y np.arrays
    """
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]

    return X, Y


def load(mode: str = "train", model_name: str = None) -> Any:
    """
    Tries to load X and Y files for training, testing or validation
    If it is a model, loads the model

    Args:
        mode (str): Mode name such as train, test, validation
        model_name (str): Saved model file's name, unless trying to load a model, leave it None
    Returns:
        Any: Returns X and Y np.arrays for datas, or a model
    """
    if not model_name:
        X, Y = [], []
        if os.path.exists('models/' + mode + "_X.npy"):
            if mode == "train":
                X = _load(mode=mode, is_X=True)
                Y = _load(mode=mode, is_X=False)
            elif mode == "validation":
                X = _load(mode=mode, is_X=True)
                Y = _load(mode=mode, is_X=False)
            elif mode == "test":
                X = _load(mode=mode, is_X=True)
                Y = _load(mode=mode, is_X=False)
        else:
            raise FileNotFoundError(f"Specified {mode} files cannot be found")
        return np.array(X, dtype=float), np.array(Y, dtype=int)

    else:
        if os.path.exists('models/' + f"{model_name}.npy"):
            model_params = _load(mode=mode, model_name=model_name)
            model_params = model_params.item()
        else:
            raise FileNotFoundError(f"Specified {mode} files cannot be found")
        return model_params


def _load(mode: str = "train", is_X: bool = True, model_name: str = None) -> np.array:
    """
    Helper function for load() function
    Makes the actual loading with numpy

    Args:
        mode (str): Mode name such as train, test, validation
        is_X (bool): Specifies if the data is X or Y
        model_name (str): Saved model file's name, unless trying to load a model, leave it None
    Returns:
        np.array: Returns the read data
    """
    path = os.path.join(os.getcwd(), "models")
    if not model_name:
        if is_X:
            path = os.path.join(path, mode + "_X.npy")
        else:
            path = os.path.join(path, mode + "_Y.npy")
    else:
        path = os.path.join(path, f"{model_name}.npy")

    with open(path, "rb") as f:
        return np.load(f, allow_pickle=True)


def load_model(model_name: str) -> Any:
    """
    Loads a model

    Args:
        mode (str): Mode name such as train, test, validation
        is_X (bool): Specifies if the data is X or Y
        model_name (str): Saved model file's name, unless trying to load a model, leave it None
    Returns:
        np.array: Returns the read data
    """
    load_model_params = read(mode="model", model_name=model_name)
    if load_model_params:
        return load_model_params


def save(arr: Any, name: str = "model", is_model: bool = False) -> None:
    """
    Saves the read data for later usages
    Or a trained model to load up later
    Saves in .npy format

    Args:
        arr (Any): Either a np.array for data or Model
        name (str): The name for the save
        is_model (bool): Specifies if the data given is a model

    Returns:
    """
    if not os.path.exists("models"):
        os.makedirs("models")

    if is_model:
        features = {
            "params": arr[0].params,
            "num_epoch": arr[0].num_epoch,
            "batch_size": arr[1],
            "activation_func": arr[2],
            "learning_rate": arr[3],
            "layer_set": arr[4]
        }
        arr = features

    np.save(f"models/{name}.npy", arr)


def get_predictions(data: np.array) -> np.array:
    """
    Return the predictions
    If there is a negative prediction, makes it equal to 0
    Args:
        data (np.array): The predictions which are made by the model

    Returns:
        np.array: Result
    """
    return np.argmax(data, 0)


def get_accuracy(predictions, Y):
    """
    Calculates the accuracy
    Args:
        predictions (np.array): The predictions which are made by the model
        Y (np.array): The actual values from the data

    Returns:
        float: The result over 100
    """
    return 100 * np.sum(predictions == Y) / Y.size


def draw_graphs(accuracy: list, loss: list) -> None:
    """
    Draws 2 graphs for accuracy list and loss list to show the relation between them and epoch number
    Args:
        accuracy (list): The accuracy values from every epoch of a model
        loss (list): The loss values from every epoch of a model

    Returns:
    """
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(7, 9)
    axs[0].plot(list(range(len(loss))), loss)
    axs[0].set(xlabel="Epoch", ylabel="Average Training Loss")

    axs[1].plot(list(range(len(accuracy))), accuracy)
    axs[1].set(xlabel="Epoch", ylabel="Average Training Accuracy")

    plt.subplots_adjust(wspace=0.01, hspace=0.4)


def visualize(weights, x, y):
    W, D = weights.shape
    W = int(round(np.sqrt(W)))
    reshaped_weights = weights.reshape(W, W, 1, D)

    fig, axs = plt.subplots(x, y)
    fig.set_size_inches(18, 18)
    index = 0
    for i in range(x):
        for j in range(y):
            axs[i, j].imshow(reshaped_weights[:, :, :, index], interpolation='nearest', cmap='gray')
            axs[i, j].axis('off')
            index += 1
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
