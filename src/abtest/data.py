"""Загрузка данных эксперимента Hillstrom (MineThatData, 2008)."""
from pathlib import Path

import pandas as pd

GROUPS = {"No E-Mail": "control", "Mens E-Mail": "mens", "Womens E-Mail": "womens"}


def load(path: str | Path = "data/hillstrom.csv") -> pd.DataFrame:
    """Читает CSV и добавляет колонку group: control / mens / womens."""
    df = pd.read_csv(path)
    df["group"] = df["segment"].map(GROUPS)
    if df["group"].isna().any():
        raise ValueError(f"Неизвестные сегменты: {df.loc[df['group'].isna(), 'segment'].unique()}")
    return df
