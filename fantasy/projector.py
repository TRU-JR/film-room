"""Projector: next-season fantasy projections, rookie projections, growth flag (R12-R14).

Method
- Players with 2+ completed seasons: weighted linear regression of fantasy points
  per game on season (weights = games played, so a 4-game season counts less),
  extrapolated one season ahead and multiplied by a 17-game season. The per-game
  value is kept between 0 and 125% of the player's best season so two-point
  lines can't extrapolate to absurd numbers.
- Players with exactly one season: per-game points x 1.10 (optimism factor) x 17.
- Players with no NFL games: no projection (NaN).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .player import Player


class Projector:
    def __init__(self, min_seasons: int = 2, optimism_factor: float = 1.10,
                 full_season_games: int = 17, target_season: int | None = None,
                 ceiling_factor: float = 1.25):
        self.min_seasons = min_seasons
        self.optimism_factor = optimism_factor
        self.full_season_games = full_season_games
        self.target_season = target_season
        self.ceiling_factor = ceiling_factor

    # ------------------------------------------------------------ internals
    def _history(self, player: Player) -> pd.DataFrame:
        hist = player.completed_seasons(before=self.target_season)
        return hist.dropna(subset=["fpts_pg"])

    def _target(self, hist: pd.DataFrame) -> int:
        return self.target_season or int(hist["season"].max()) + 1

    def _fit(self, hist: pd.DataFrame) -> tuple[float, float]:
        x = hist["season"].to_numpy(dtype=float)
        y = hist["fpts_pg"].to_numpy(dtype=float)
        w = hist["games"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
        return float(slope), float(intercept)

    # --------------------------------------------------------------- public
    def slope(self, player: Player) -> float:
        """Change in fantasy points per game per season (0 if under min_seasons)."""
        hist = self._history(player)
        if len(hist) < self.min_seasons:
            return 0.0
        return round(self._fit(hist)[0], 3)

    def is_growing(self, player: Player) -> bool:
        return self.slope(player) > 0

    def _pg_from_hist(self, hist: pd.DataFrame) -> float:
        if hist.empty or hist["games"].sum() == 0:
            return math.nan
        if len(hist) < self.min_seasons:
            return round(float(hist["fpts_pg"].iloc[-1]) * self.optimism_factor, 2)
        slope, intercept = self._fit(hist)
        pred = intercept + slope * self._target(hist)
        ceiling = float(hist["fpts_pg"].max()) * self.ceiling_factor
        return round(float(np.clip(pred, 0.0, ceiling)), 2)

    def project_per_game(self, player: Player) -> float:
        return self._pg_from_hist(self._history(player))

    def project(self, player: Player) -> float:
        """Projected full-season fantasy points for the target season."""
        pg = self.project_per_game(player)
        return round(pg * self.full_season_games, 1) if not math.isnan(pg) else math.nan

    def method(self, player: Player) -> str:
        n = len(self._history(player))
        if n == 0:
            return "No NFL games"
        return "Regression" if n >= self.min_seasons else "One-season x 1.10"

    def likely_range(self, player: Player) -> tuple[float, float]:
        """Low/high season totals: projection +/- one std. dev. of past per-game seasons."""
        hist = self._history(player)
        pg = self.project_per_game(player)
        if math.isnan(pg):
            return (math.nan, math.nan)
        spread = float(hist["fpts_pg"].std(ddof=0)) if len(hist) > 1 else pg * 0.25
        spread = max(spread, pg * 0.10)
        g = self.full_season_games
        return (round(max(pg - spread, 0) * g, 1), round((pg + spread) * g, 1))

    def blended(self, player: Player, current: pd.Series | None, remaining_games: int) -> float:
        """Season total that mixes in the current season so far.

        actual points so far + remaining games x weighted per-game rate, where the
        current pace gets weight g / (g + 4) after g games.
        """
        base = self.project_per_game(player)
        if current is None or current.get("games", 0) == 0:
            return round(base * (remaining_games + 0), 1) if not math.isnan(base) else math.nan
        g = float(current["games"])
        pace = float(current["fpts_pg"])
        w = g / (g + 4.0)
        rate = pace if math.isnan(base) else w * pace + (1 - w) * base
        return round(float(current["fpts"]) + rate * remaining_games, 1)

    def projection_table(self, players: list[Player]) -> pd.DataFrame:
        rows = []
        g = self.full_season_games
        for p in players:
            hist = self._history(p)
            pg = self._pg_from_hist(hist)
            if math.isnan(pg):
                continue
            n = len(hist)
            slope = round(self._fit(hist)[0], 3) if n >= self.min_seasons else 0.0
            spread = float(hist["fpts_pg"].std(ddof=0)) if n > 1 else pg * 0.25
            spread = max(spread, pg * 0.10)
            rows.append({"player_id": p.player_id, "name": p.name, "position": p.position,
                         "team": p.team, "seasons_used": n,
                         "method": "Regression" if n >= self.min_seasons else "One-season x 1.10",
                         "proj_pg": pg, "projection": round(pg * g, 1),
                         "low": round(max(pg - spread, 0) * g, 1), "high": round((pg + spread) * g, 1),
                         "slope": slope, "growing": slope > 0})
        return pd.DataFrame(rows)

    def backtest(self, df: pd.DataFrame, season: int, min_games: int = 6) -> pd.DataFrame:
        """Project `season` using only earlier seasons and compare to what happened."""
        tester = Projector(self.min_seasons, self.optimism_factor, self.full_season_games,
                           target_season=season, ceiling_factor=self.ceiling_factor)
        actual = df[(df["season"] == season) & (df["games"] >= min_games)]
        rows = []
        for pid, act in actual.set_index("player_id").iterrows():
            hist = df[(df["player_id"] == pid) & (df["season"] < season)]
            if hist.empty:
                continue
            p = Player(pid, act["name"], act["position"], act["team"], hist)
            pred = tester.project_per_game(p)
            if math.isnan(pred):
                continue
            rows.append({"player_id": pid, "name": act["name"], "position": act["position"],
                         "method": tester.method(p), "projected_pg": pred,
                         "actual_pg": float(act["fpts_pg"]),
                         "error_pg": round(pred - float(act["fpts_pg"]), 2)})
        return pd.DataFrame(rows)
