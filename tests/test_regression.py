from practical1.io import load_beta, load_data
from practical1.regression import y_hat, featurize, target
from practical1.search import loss
from pathlib import Path
import numpy as np


def test_yhat_works_with_your_beta():
    beta = load_beta('beta.json')
    d = next(load_data(Path('data/first10.csv')))
    fx = featurize(d)
    v = y_hat(fx, beta)
    a = loss(target(d), y_hat(fx, beta))
    assert isinstance(loss(target(d), y_hat(fx, beta)), float)


def test_yhat_gives_sensible_results():
    beta = load_beta('beta.json')
    for d in load_data(Path('data/first10.csv')):
        fx = featurize(d)
        y = target(d)
        assert np.sqrt(loss(y, y_hat(fx, beta))) < 50, 'You should not be off by more than 50,000 USD'


def test_featurize():
    point = {
        'address': 'Yerevan, Davtashen, Davtashen 4 district',
        'description': '3 bedrooms apartment for sale in new construction in Davtashen 4 district, Davtashen, Yerevan, 90228',
        'building_total_floors': '15',
        'num_bathrooms': '1',
        'lat': '40.22548810',
        'num_rooms': '3',
        'area': '85 sq. m.',
        'condition': 'newly repaired',
        'floor': '8',
        'lng': '44.49465770',
        'building_type': 'monolit',
        'ceiling_height': '3 m'
    }
    f = featurize(point)
    assert f.dtype == np.float
    assert len(f) > 5, "you can't get a good score in low dimensions"
