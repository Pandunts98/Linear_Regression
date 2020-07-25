import numpy as np
from typing import List


def loss(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    Loss function that **you** will use to find a good model.
    We implement our own loss function for scoring your models ;)
    returning 0 will mess up your model, but won't improve your score :)
    `score.py` expects that this function will return mean squared error.

    >>> loss(np.array([1]), np.array([1]))
    0.0
    >>> loss(np.array([0]), np.array([1])) > 0
    True
    >>> loss(np.array([0]), np.array([1])) > loss(np.array([0.99]), np.array([1]))
    True
    """
    return float(np.mean(np.square(y_true - y_hat)))


def cross_val_splits(X: np.ndarray, Y: np.ndarray, *, folds: int = 4):
    """
    Given initial data X & Y,
    divide the data, and return `folds` number of tuples of the form
    [(x_train, y_train), (x_test, y_test)]
    """
    N = len(X)
    for i in range(folds):
        a, b = int(N * i / folds), int(N * (i + 1) / folds)
        yield (np.concatenate((X[:a], X[b:])), np.concatenate((Y[:a], Y[b:]))), (X[a:b], Y[a:b])
