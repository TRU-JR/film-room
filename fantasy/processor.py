"""StatProcessor: adds derived columns to the master player-season table (R3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import ScoringFormat

PER_GAME_STATS = [
    "passing_yards", "passing_tds", "passing_interceptions", "attempts", "completions",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
]

AGE_BINS = [0, 23, 26, 29, 100]
AGE_LABELS = ["23 and under", "24–26", "27–29", "30 and over"]

# Counting stats that can be summed across seasons
COUNTING_STATS = [
    "games", "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "sacks_suffered", "passing_air_yards", "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
    "receiving_yards_after_catch", "special_teams_tds", "fumbles_lost", "two_pt",
]


def safe_div(num: pd.Series, den: pd.Series, digits: int = 2) -> pd.Series:
    """num / den with NaN wherever den is zero."""
    den = pd.to_numeric(den, errors="coerce").astype(float)
    return (pd.to_numeric(num, errors="coerce").astype(float) / den.where(den != 0, np.nan)).round(digits)


class StatProcessor:
    """Mutates `df` in place; every add_* method is safe to call more than once."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_age(self) -> None:
        """Age (years, one decimal) on September 1 of each season."""
        birth = pd.to_datetime(self.df["birth_date"], errors="coerce")
        kickoff = pd.to_datetime(self.df["season"].astype(int).astype(str) + "-09-01")
        self.df["age"] = ((kickoff - birth).dt.days / 365.25).round(1)

    def add_age_bins(self) -> None:
        if "age" not in self.df:
            self.add_age()
        self.df["age_bin"] = pd.cut(self.df["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
        self.df.loc[self.df["age"].isna(), "age_bin"] = "Unknown"

    def add_per_game(self) -> None:
        for col in PER_GAME_STATS:
            if col in self.df:
                self.df[f"{col}_pg"] = safe_div(self.df[col], self.df["games"], 2)
        self.df["yards_per_carry"] = safe_div(self.df["rushing_yards"], self.df["carries"], 2)
        self.df["yards_per_target"] = safe_div(self.df["receiving_yards"], self.df["targets"], 2)
        self.df["catch_rate"] = safe_div(self.df["receptions"], self.df["targets"], 3)
        self.df["yards_per_attempt"] = safe_div(self.df["passing_yards"], self.df["attempts"], 2)
        self.df["total_tds"] = self.df["passing_tds"] + self.df["rushing_tds"] + self.df["receiving_tds"]
        self.df["scrimmage_yards"] = self.df["rushing_yards"] + self.df["receiving_yards"]
        self.df["opportunities"] = self.df["carries"] + self.df["targets"]
        self.df["opportunities_pg"] = safe_div(self.df["opportunities"], self.df["games"], 2)

    def add_fantasy_points(self, scoring: ScoringFormat) -> None:
        self.df["fpts"] = scoring.points(self.df)
        self.df["fpts_pg"] = safe_div(self.df["fpts"], self.df["games"], 2)
        self.df["scoring"] = scoring.name

    def run_all(self, scoring: ScoringFormat) -> pd.DataFrame:
        self.add_age()
        self.add_age_bins()
        self.add_per_game()
        self.add_fantasy_points(scoring)
        return self.df


def aggregate_players(df: pd.DataFrame, scoring: ScoringFormat) -> pd.DataFrame:
    """Collapse player-season rows into one row per player (multi-season totals)."""
    if df.empty:
        return df.copy()
    g = df.sort_values("season").groupby("player_id", as_index=False)
    cols = [c for c in COUNTING_STATS if c in df.columns]
    totals = g[cols].sum()
    info = g.agg(name=("name", "last"), position=("position", "last"), team=("team", "last"),
                 season=("season", "last"), birth_date=("birth_date", "last"),
                 first_season=("season", "min"), seasons=("season", "nunique"))
    out = info.merge(totals, on="player_id")
    if "target_share" in df:
        out = out.merge(g["target_share"].mean().rename(columns={"target_share": "target_share"}), on="player_id")
    proc = StatProcessor(out)
    return proc.run_all(scoring)
