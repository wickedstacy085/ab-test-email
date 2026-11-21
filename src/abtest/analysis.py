"""Оценка эффекта: доли, средние, бутстрап, CUPED, поправка Холма."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest


def _split(df: pd.DataFrame, metric: str, treat: str, control: str):
    x = df.loc[df["group"] == treat, metric].to_numpy(float)
    y = df.loc[df["group"] == control, metric].to_numpy(float)
    return x, y


def compare_proportion(df: pd.DataFrame, metric: str, treat: str, control: str = "control",
                       alpha: float = 0.05) -> dict:
    """Разность долей (treat − control): z-тест и ДИ Вальда."""
    x, y = _split(df, metric, treat, control)
    p1, p0 = x.mean(), y.mean()
    diff = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / len(x) + p0 * (1 - p0) / len(y))
    z = stats.norm.ppf(1 - alpha / 2)
    _, p = proportions_ztest([x.sum(), y.sum()], [len(x), len(y)])
    return {"lift_abs": diff, "lift_rel": diff / p0, "ci_low": diff - z * se,
            "ci_high": diff + z * se, "p_value": float(p)}


def compare_mean(df: pd.DataFrame, metric: str, treat: str, control: str = "control",
                 alpha: float = 0.05) -> dict:
    """Разность средних (treat − control): t-тест Уэлча и ДИ."""
    x, y = _split(df, metric, treat, control)
    diff = x.mean() - y.mean()
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    dof = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    t = stats.t.ppf(1 - alpha / 2, dof)
    p = stats.ttest_ind(x, y, equal_var=False).pvalue
    base = y.mean()
    return {"lift_abs": diff, "lift_rel": diff / base if base else np.nan,
            "ci_low": diff - t * se, "ci_high": diff + t * se, "p_value": float(p)}


def bootstrap_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 10_000, seed: int = 42,
                   alpha: float = 0.05) -> tuple[float, float, float]:
    """Разность средних и перцентильный бутстрап-ДИ (выборки ресэмплируются независимо)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.empty(n_boot)
    chunk = 500  # по частям, чтобы не держать в памяти матрицу n_boot × n
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        bx = x[rng.integers(0, len(x), (k, len(x)))].mean(axis=1)
        by = y[rng.integers(0, len(y), (k, len(y)))].mean(axis=1)
        d[s:s + k] = bx - by
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float(x.mean() - y.mean()), float(lo), float(hi)


def cuped(df: pd.DataFrame, metric: str, covariate: str = "history") -> pd.Series:
    """Y − θ(X − X̄), θ = cov(Y, X)/var(X) по всем группам вместе. X должен быть измерен до теста."""
    y, x = df[metric].astype(float), df[covariate].astype(float)
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean())


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Поправка Холма на множественные сравнения."""
    keys = list(p_values)
    adj = multipletests([p_values[k] for k in keys], method="holm")[1]
    return {k: float(v) for k, v in zip(keys, adj)}
