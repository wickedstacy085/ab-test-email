"""Дизайн эксперимента: размер выборки, минимальный детектируемый эффект, проверка SRM."""
import math

from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def sample_size_proportion(p0: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Размер каждой группы, чтобы заметить рост доли с p0 до p0 + mde (двусторонний тест)."""
    h = proportion_effectsize(p0 + mde, p0)
    return math.ceil(NormalIndPower().solve_power(h, alpha=alpha, power=power, ratio=1.0))


def mde_proportion(p0: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Минимальный абсолютный прирост доли, который тест заметит при n наблюдений в группе."""
    h = NormalIndPower().solve_power(nobs1=n_per_group, alpha=alpha, power=power, ratio=1.0)
    # обратное к h = 2·asin(√p1) − 2·asin(√p0)
    p1 = math.sin(math.asin(math.sqrt(p0)) + h / 2) ** 2
    return p1 - p0


def srm_test(counts: dict[str, int], expected_shares: dict[str, float]) -> tuple[float, float]:
    """χ²-тест на sample ratio mismatch: совпадает ли фактическое деление с задуманным."""
    keys = list(counts)
    total = sum(counts.values())
    observed = [counts[k] for k in keys]
    expected = [expected_shares[k] * total for k in keys]
    chi2, p = stats.chisquare(observed, expected)
    return float(chi2), float(p)
