import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from abtest.design import mde_proportion, sample_size_proportion, srm_test


def test_sample_size_matches_formula():
    n = sample_size_proportion(0.10, 0.01)
    assert 14_700 <= n <= 14_800
    h = proportion_effectsize(0.11, 0.10)
    assert n == pytest.approx(NormalIndPower().solve_power(h, alpha=0.05, power=0.8), abs=1)


def test_mde_inverts_sample_size():
    n = sample_size_proportion(0.10, 0.01)
    assert mde_proportion(0.10, n) == pytest.approx(0.01, abs=1e-3)


def test_srm_balanced_passes():
    assert srm_test({"a": 500, "b": 500}, {"a": .5, "b": .5})[1] > 0.99


def test_srm_imbalanced_fails():
    assert srm_test({"a": 550, "b": 450}, {"a": .5, "b": .5})[1] < 0.01
