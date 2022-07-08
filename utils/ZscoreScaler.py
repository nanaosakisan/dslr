import numpy as np


class ZscoreScaler:
    def __init__(self, mean=None, std=None):
        if mean and std:
            self.mean = mean
            self.std = std

    def fit(self, X: np.ndarray):
        print("fit")
        print(X.shape)
        print(X)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X: np.ndarray):
        return (X - self.mean) / self.std

    def reverse(self, X: np.ndarray):
        return X * self.std + self.mean

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)
