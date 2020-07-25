from practical1 import search
import numpy as np


def test_cross_val_splits_outputs_correct_format():
    X = np.arange(10)
    Y = np.ones(10)

    for [(x, y), (tx, ty)] in search.cross_val_splits(X, Y, folds=2):
        assert len(x) == len(set(x))
        assert len(tx) == len(set(tx))


def test_cross_val_splits_uses_all_data():
    X = np.arange(10)
    Y = np.ones(10)

    all_train_x, all_test_x = set(), set()
    for (x, y), (tx, ty) in search.cross_val_splits(X, Y, folds=5):
        all_train_x |= set(x)
        all_test_x |= set(tx)

        assert len(x) == 8
        assert len(tx) == 2

        assert len(set(x) & set(tx)) == 0

    assert len(all_test_x) == 10
    assert len(all_train_x) == 10
