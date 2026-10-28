"""ScoringFormat: converts raw box-score stats into fantasy points (R3, R15)."""
from __future__ import annotations

import pandas as pd

# Weights shared by every format; only the reception value changes.
_BASE_WEIGHTS: dict[str, float] = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "passing_interceptions": -2.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "fumbles_lost": -2.0,
    "two_pt": 2.0,
    "special_teams_tds": 6.0,
}

RECEPTION_VALUE = {"PPR": 1.0, "Half-PPR": 0.5, "Standard": 0.0}

# Team defense / special teams scoring (identical in every format).
DST_WEIGHTS: dict[str, float] = {
    "def_sacks": 1.0,
    "def_interceptions": 2.0,
    "fumble_recovery_opp": 2.0,
    "def_tds": 6.0,
    "def_safeties": 2.0,
    "special_teams_tds": 6.0,
    "blocked_kicks": 2.0,
}

# (upper bound of points allowed, fantasy points) - checked in order
POINTS_ALLOWED_TIERS = [(0, 10), (6, 7), (13, 4), (20, 1), (27, 0), (34, -1), (10_000, -4)]


def points_allowed_score(points_allowed: float) -> float:
    """Fantasy points a defense earns for holding an offense to `points_allowed`."""
    for upper, pts in POINTS_ALLOWED_TIERS:
        if points_allowed <= upper:
            return float(pts)
    return -4.0


class ScoringFormat:
    """A named scoring system expressed as stat -> points weights."""

    def __init__(self, name: str = "PPR", weights: dict[str, float] | None = None):
        if weights is None:
            if name not in RECEPTION_VALUE:
                raise ValueError(f"Unknown scoring format '{name}'. Use PPR, Half-PPR or Standard.")
            weights = dict(_BASE_WEIGHTS, receptions=RECEPTION_VALUE[name])
        self.name = name
        self.weights = weights

    @classmethod
    def all_formats(cls) -> list[str]:
        return list(RECEPTION_VALUE)

    def points(self, df: pd.DataFrame) -> pd.Series:
        """Fantasy points for every row of `df`. Missing stat columns count as zero."""
        total = pd.Series(0.0, index=df.index)
        for col, weight in self.weights.items():
            if col in df.columns:
                total = total + pd.to_numeric(df[col], errors="coerce").fillna(0.0) * weight
        return total.round(2)

    def __repr__(self) -> str:
        return f"ScoringFormat({self.name!r})"
