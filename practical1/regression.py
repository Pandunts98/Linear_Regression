import argparse
import numpy as np
from typing import List
import itertools
import matplotlib.pyplot as plt


def target(apartment: dict) -> float:
    """
    Return the variable we want to use to predict, the apartment price
    divided by 1000.
    We divide by 1000, to make numbers look less scary :)
    >>> target({'address': 'Yerevan', 'price': '102,000'})
    102.0
    """
    return float(apartment['price'][:-4])


def featurize(apartment: dict) -> np.ndarray:
    """
    :param apartment: a dictionary containing the data
    :return: feature vector for that point - np.ndarray
    """
    lat, lng = 111, 85  # Degree precision versus length km
    center = [40.177200, 44.503490]
    distance = (np.sqrt(sum((np.array([float(apartment['lat']) * lat, float(apartment['lng']) * lng]) - np.array(
        [center[0] * lat, center[1] * lng])) ** 2)))
    area = float(apartment['area'].strip(' sq. m.'))
    rooms = float(apartment['num_rooms'][0]) if apartment['num_rooms'] != '' else 7.
    bathrooms = float(apartment['num_bathrooms'][0]) if apartment['num_bathrooms'] != '' else 5.
    floor = float(apartment['floor'])
    total_floor = float(apartment["building_total_floors"])
    floors = total_floor + floor
    floors = 6 + (6 - floors) if floors < 6 else floors
    height = float(apartment['ceiling_height'].split(' ', 1)[0])
    height = 1.5 if height > 3 else height
    if apartment['building_type'] == 'stone':
        build_type = 1.5
    elif apartment['building_type'] == 'panel':
        build_type = 2.5
    elif apartment['building_type'] == 'monolit':
        build_type = 0.5
    else:
        build_type = 3.
    if apartment['condition'] == 'good':
        condition = 3.
    elif apartment['condition'] == 'newly repaired':
        condition = 4.
    else:
        condition = 1.
    return np.array([1, area ** 0.5, area, distance, area * condition, area * build_type,
                     rooms * area, bathrooms, height, floors])


def fit_linear_regression(X: np.ndarray, Y: np.ndarray, *, l: float = 0) -> np.ndarray:
    """
    Fit linear regression to the data
    >>> fit_linear_regression(np.array([[1, 1], [1, 6]]), np.array([1, 2]))
    array([0.8, 0.2])
    """
    return np.linalg.pinv(X.T.dot(X) + l * np.eye(X.shape[1])).dot(X.T).dot(Y)


def y_hat(x: np.ndarray, beta: np.ndarray) -> float or np.ndarray:
    """
    Note: This is different from homework 1.
    Here we are fitting linear regression! And we want to make y_hat to
    work both when x is a single point, and when x is a matrix!
    :param x: input vector or matrix
    :param beta: model parameters
    :return: prediction(s)
    >>> y_hat(np.array([1, 1]), np.array([0.1, 2.0]))
    2.1
    >>> y_hat(np.array([[1, 1], [2, 3]]), np.array([0.1, 2.0]))
    array([2.1, 6.2])
    """
    return x.T.dot(beta) if x.ndim == 1 else np.array([i.T.dot(beta) for i in x])
