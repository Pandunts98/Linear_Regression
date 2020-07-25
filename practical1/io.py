from typing import Iterable
import csv
import json
import numpy as np
import gzip


def load_data(path) -> Iterable[dict]:
    if path.suffix == '.csv':
        with open(path, 'r') as infile:
            reader = csv.DictReader(infile)
            yield from reader
    elif path.suffix == '.gz':
        with gzip.open(path, 'rt') as infile:
            reader = csv.DictReader(infile)
            yield from reader


def export_beta(path, beta):
    with open(path, 'w') as outfile:
        json.dump(beta.tolist(), outfile)


def load_beta(path):
    with open(path, 'r') as infile:
        return np.array(json.load(infile))
