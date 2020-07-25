import argparse
from pathlib import Path
import json
import numpy as np
from save_beta import load_data
from practical1.regression import target, featurize, y_hat
from practical1.search import loss
from practical1.io import load_beta


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/first10.csv', type=Path,
                        help='CSV file with the apartment data.')
    parser.add_argument('--input-beta', default='beta.json',
                        help='json file to save the beta in.')
    args = parser.parse_args(*argument_array)
    return args


if __name__ == '__main__':
    args = parse_args()
    raw_data = list(load_data(args.data))
    data = [d for d in raw_data if 300000 > float(d['price'].replace(',', '')) > 30000]
    X = np.array([featurize(x) for x in data])
    Y = np.array([target(x) for x in data])
    beta = load_beta(args.input_beta)
    print(f'Mean Error = {np.sqrt(loss(Y, y_hat(X, beta))):,.3f}')
