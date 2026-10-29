import pytest

from src.house_price.data import FEATURES, TARGET, make_housing_data


def test_data_shape_and_columns():
    frame = make_housing_data(n_samples=100, random_state=7)
    assert len(frame) == 100
    assert list(frame.columns) == FEATURES + [TARGET]
    assert frame.notna().all().all()


def test_data_is_reproducible():
    assert make_housing_data(50, 9).equals(make_housing_data(50, 9))


def test_rejects_too_small_dataset():
    with pytest.raises(ValueError):
        make_housing_data(10)
