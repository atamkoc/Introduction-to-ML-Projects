import utility
import numpy as np


class KNN:
    def __init__(self, k: int):
        self.k = k
        self.train_x = None
        self.train_y = None

    def fit(self, x, y):
        self.train_x = x
        self.train_y = y