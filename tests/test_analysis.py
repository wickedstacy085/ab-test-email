import numpy as np
import pandas as pd
import pytest
from conftest import DATA

import abtest
from abtest.analysis import bootstrap_diff, compare_mean, compare_proportion, cuped, holm


def _synthetic(n=5000, effect=0.1, rho=0.6, seed=1):
    rng = np.random.default_rng(seed)
    cov = rng.standard_normal(2 * n)
    noise = rng.standard_normal(2 * n)
    y = rho * cov + np.sqrt(1 - rho**2) * noise
    group = np.repeat(["control", "t"], n)
    y[n:] += effect
    return pd.DataFrame({"y": y, "history": cov, "group": group})


def test_bootstrap_matches_t_interval():
    df = _synthetic(rho=0.0)
    x = df.loc[df.group == "t", "y"].to_numpy()
    y = df.loc[df.group == "control", "y"].to_numpy()
    _, lo, hi = bootstrap_diff(x, y)
    t = compare_mean(df, "y", "t")
    assert abs(lo - t["ci_low"]) < 0.01
    assert abs(hi - t["ci_high"]) < 0.01


def test_cuped_reduces_variance_keeps_effect():
    df = _synthetic()
    raw = compare_mean(df, "y", "t")["lift_abs"]
    df["y_cuped"] = cuped(df, "y")
    adj = compare_mean(df, "y_cuped", "t")["lift_abs"]
    assert df["y_cuped"].var() <= 0.7 * df["y"].var()
    assert abs(adj - raw) < 0.1 * abs(raw)


def test_holm():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_mens_email_raises_visits():
    r = compare_proportion(abtest.load(DATA), "visit", "mens")
    assert r["lift_abs"] > 0
    assert r["p_value"] < 1e-10
    assert r["ci_low"] < r["lift_abs"] < r["ci_high"]
