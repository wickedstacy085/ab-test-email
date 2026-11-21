"""Эффект рассылки по сегментам клиентов."""
import pandas as pd

from .analysis import compare_mean, compare_proportion, holm


def effect_by_segment(df: pd.DataFrame, metric: str, treat: str, by: str,
                      control: str = "control") -> pd.DataFrame:
    """Эффект treat против control внутри каждого значения `by`; p-value с поправкой Холма.

    Для бинарной метрики — разность долей, для непрерывной — разность средних (Уэлч)."""
    binary = set(df[metric].unique()) <= {0, 1}
    compare = compare_proportion if binary else compare_mean
    rows = {}
    for value, part in df[df["group"].isin([treat, control])].groupby(by):
        r = compare(part, metric, treat, control)
        rows[value] = {"n": len(part), **r}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["p_holm"] = pd.Series(holm(out["p_value"].to_dict()))
    return out
