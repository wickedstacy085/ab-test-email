import pytest
from conftest import DATA
import abtest


@pytest.fixture(scope="module")
def df():
    return abtest.load(DATA)


def test_load(df):
    assert len(df) == 64_000
    counts = df["group"].value_counts()
    assert set(counts.index) == {"control", "mens", "womens"}
    assert counts.between(21_300, 21_400).all()
    assert (df["spend"] >= 0).all()
    assert (df["conversion"] <= df["visit"]).all()
