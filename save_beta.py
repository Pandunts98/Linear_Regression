import argparse
import numpy as np
import csv
import json
from collections import defaultdict
from pathlib import Path

from practical1.regression import target, featurize
from practical1.regression import fit_linear_regression, y_hat
from practical1.search import cross_val_splits, loss
from practical1.io import load_data, export_beta


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path,
                        default=Path('data/yerevan_may_2020.csv.gz'),
                        help='CSV file with the apartment data.')
    parser.add_argument('--output-beta', default='beta.json',
                        help='json file to save the beta in.')
    args = parser.parse_args(*argument_array)
    return args


def find_beta(X, Y) -> np.ndarray:
    res = {}
    for l in np.logspace(-3, 3, 13):
        error = []
        for (x, y), (tx, ty) in cross_val_splits(X, Y, folds=5):
            error.append((loss(y_hat(tx, fit_linear_regression(x, y, l=l)), ty)))
        res[l] = sum(error) / len(error)
    lam = min(res.keys(), key=(lambda k: res[k]))
    return fit_linear_regression(X, Y, l=lam)


if __name__ == '__main__':
    args = parse_args()
    raw_data = list(load_data(args.data))
    data = [d for d in raw_data if 300000 > float(d['price'].replace(',', '')) > 30000]
    X = np.array([featurize(x) for x in data])
    Y = np.array([target(x) for x in data])
    print(X.shape, Y.shape)
    beta = find_beta(X, Y)
    export_beta(args.output_beta, beta)
