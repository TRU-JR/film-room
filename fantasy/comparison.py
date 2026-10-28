"""Comparison: side-by-side players (max six, same position) and team units (R8, R9)."""
from __future__ import annotations

import pandas as pd

from .errors import ComparisonError
from .player import Player
from .teams import DEFENSE_STATS, OFFENSE_STATS, team_season_table

MAX_PLAYERS = 6


class Comparison:
    def __init__(self, stat: str = "fpts", unit: str = "Offense"):
        self.players: list[Player] = []
        self.stat = stat
        self.teams: list[str] = []
        self.unit = unit

    def add_player(self, p: Player) -> None:
        if p in self.players:
            raise ComparisonError(f"{p.name} is already in the comparison.")
        if len(self.players) >= MAX_PLAYERS:
            raise ComparisonError(f"You can compare up to {MAX_PLAYERS} players. Remove one to add {p.name}.")
        if self.players and p.position != self.players[0].position:
            raise ComparisonError(
                f"{p.name} is a {p.position}; this comparison is for {self.players[0].position}s. "
                "Compare players from one position group at a time.")
        self.players.append(p)

    def compare_players(self) -> pd.DataFrame:
        """Season x player table of the chosen stat, plus a career row."""
        if not self.players:
            return pd.DataFrame()
        frames = []
        for p in self.players:
            if self.stat not in p.seasons:
                raise ComparisonError(f"'{self.stat}' isn't tracked for {p.name}.")
            s = p.seasons.set_index("season")[self.stat].rename(p.name)
            frames.append(s)
        table = pd.concat(frames, axis=1).sort_index()
        table.index = table.index.astype(str)
        totals = {}
        for p in self.players:
            if self.stat.endswith("_pg") or self.stat in ("age", "target_share", "carry_share",
                                                          "catch_rate", "yards_per_carry",
                                                          "yards_per_target"):
                totals[p.name] = round(p.seasons[self.stat].mean(), 2)
            else:
                totals[p.name] = p.seasons[self.stat].sum()
        table.loc["Career"] = pd.Series(totals)
        return table.astype(float)

    def compare_teams(self, team_games: pd.DataFrame, season: int,
                      player_weeks: pd.DataFrame | None = None) -> pd.DataFrame:
        table = team_season_table(team_games, season, self.unit, player_weeks)
        if self.teams:
            table = table[table["team"].isin(self.teams)]
        return table

    @property
    def stat_options(self) -> dict[str, str]:
        return OFFENSE_STATS if self.unit == "Offense" else DEFENSE_STATS
