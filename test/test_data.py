import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data_preparation import load_data, validate_data


def test_load_data():
    df = load_data()
    assert df is not None
    assert len(df) > 0


def test_validate_data():
    df = load_data()
    rep = validate_data(df)
    assert 'shape' in rep
    assert rep['shape'][0] > 0
