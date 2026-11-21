from conftest import DATA

import abtest
from abtest.segments import effect_by_segment


def test_effect_by_channel():
    df = abtest.load(DATA)
    t = effect_by_segment(df, "visit", "womens", "channel")
    assert set(t.index) == {"Phone", "Web", "Multichannel"}
    assert t["n"].sum() == (df["group"].isin(["womens", "control"])).sum()
    assert {"lift_abs", "ci_low", "ci_high", "p_holm"} <= set(t.columns)
    assert (t["p_holm"] <= 1).all()
