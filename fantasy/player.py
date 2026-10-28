"""Player: one player's career, built from the processed player-season table (R7)."""
from __future__ import annotations

import pandas as pd

from .processor import COUNTING_STATS


class Player:
    def __init__(self, player_id: str, name: str, position: str, team: str,
                 seasons: pd.DataFrame | None = None):
        self.player_id = player_id
        self.name = name
        self.position = position
        self.team = team
        if seasons is None or len(seasons) == 0:
            self.seasons = pd.DataFrame(columns=["season", "games", "fpts", "fpts_pg"])
        else:
            self.seasons = seasons.sort_values("season").reset_index(drop=True)

    @classmethod
    def from_df(cls, df: pd.DataFrame, player_id: str, team: str | None = None) -> "Player":
        rows = df[df["player_id"] == player_id]
        if rows.empty:
            raise KeyError(f"No stats found for player id {player_id}")
        last = rows.sort_values("season").iloc[-1]
        return cls(player_id, last["name"], last["position"], team or last["team"], rows)

    # ------------------------------------------------------------------ views
    def career_totals(self) -> pd.Series:
        cols = [c for c in COUNTING_STATS + ["fpts"] if c in self.seasons]
        totals = self.seasons[cols].sum()
        games = totals.get("games", 0)
        totals["fpts_pg"] = round(totals["fpts"] / games, 2) if games else 0.0
        return totals

    def season(self, year: int) -> pd.Series:
        rows = self.seasons[self.seasons["season"] == year]
        if rows.empty:
            raise KeyError(f"{self.name} has no stats for {year}")
        return rows.iloc[0]

    def per_game(self) -> pd.DataFrame:
        cols = ["season"] + [c for c in self.seasons.columns if c.endswith("_pg")]
        return self.seasons[cols]

    def num_seasons(self) -> int:
        return int((self.seasons["games"] > 0).sum())

    def completed_seasons(self, before: int | None = None) -> pd.DataFrame:
        s = self.seasons[self.seasons["games"] > 0]
        if "partial" in s:
            s = s[~s["partial"].astype(bool)]
        if before is not None:
            s = s[s["season"] < before]
        return s

    @property
    def label(self) -> str:
        return f"{self.name} ({self.position}, {self.team})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Player) and other.player_id == self.player_id

    def __hash__(self) -> int:
        return hash(self.player_id)

    def __repr__(self) -> str:
        return f"Player({self.name!r}, {self.position}, {self.team})"


def build_players(df: pd.DataFrame, teams: dict[str, str] | None = None) -> dict[str, Player]:
    """Every player in `df`, keyed by player_id (one groupby instead of one filter per player)."""
    out: dict[str, Player] = {}
    for pid, rows in df.groupby("player_id", sort=False):
        rows = rows.sort_values("season")
        last = rows.iloc[-1]
        team = (teams or {}).get(pid, last["team"])
        out[pid] = Player(pid, last["name"], last["position"], team, rows)
    return out
