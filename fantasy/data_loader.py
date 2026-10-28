"""DataLoader: retrieves NFL data from nflverse and caches it locally (R1, R2).

The first run downloads weekly player stats, weekly team stats, the schedule,
current rosters and player birth dates, trims them to what the app needs and
writes small CSVs into `cache_dir`. Later runs read the CSVs, so the app starts
in a second or two. `refresh()` re-downloads everything (e.g. after a new week).
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

PLAYER_STATS = [
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "sacks_suffered", "passing_air_yards", "passing_epa",
    "carries", "rushing_yards", "rushing_tds", "rushing_epa",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa",
    "special_teams_tds", "fumbles_lost", "two_pt", "nflverse_ppr",
]

TEAM_STATS = [
    "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
    "sacks_suffered", "carries", "rushing_yards", "rushing_tds", "targets",
    "fumbles_lost_total", "def_sacks", "def_interceptions", "fumble_recovery_opp",
    "def_tds", "def_safeties", "special_teams_tds", "blocked_kicks",
]

CACHE_FILES = {
    "player_weeks": "player_weeks.csv",
    "player_stats": "player_stats.csv",
    "team_games": "team_games.csv",
    "schedule": "schedule.csv",
    "current_rosters": "current_rosters.csv",
}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


class DataLoader:
    """Loads and caches five completed seasons plus the season in progress."""

    def __init__(self, cache_dir: str = "data_cache", seasons: list[int] | None = None,
                 current_season: int | None = None, source: str = NFLVERSE):
        today = date.today()
        self.current_season = current_season or (today.year if today.month >= 9 else today.year - 1)
        self.seasons = seasons or list(range(self.current_season - 5, self.current_season))
        self.cache_dir = cache_dir
        self.source = source  # a URL base, or a local folder with the same file names
        self.player_weeks = pd.DataFrame()
        self.player_stats = pd.DataFrame()
        self.team_stats = pd.DataFrame()   # one row per team per game
        self.schedule = pd.DataFrame()
        self.current_rosters = pd.DataFrame()

    # ------------------------------------------------------------------ public
    def load(self) -> None:
        """Read the cache, downloading first if any cached file is missing."""
        if not all(os.path.exists(self._path(f)) for f in CACHE_FILES.values()):
            self.refresh()
            return
        self.player_weeks = pd.read_csv(self._path(CACHE_FILES["player_weeks"]))
        self.player_stats = pd.read_csv(self._path(CACHE_FILES["player_stats"]))
        self.team_stats = pd.read_csv(self._path(CACHE_FILES["team_games"]))
        self.schedule = pd.read_csv(self._path(CACHE_FILES["schedule"]))
        self.current_rosters = pd.read_csv(self._path(CACHE_FILES["current_rosters"]))
        cached = set(self.player_stats["season"].unique())
        wanted = set(self.seasons) | {self.current_season}
        if not wanted.issubset(cached):
            self.refresh()

    def refresh(self) -> None:
        """Download fresh data from nflverse and rewrite the cache."""
        all_seasons = self.seasons + [self.current_season]
        weeks = pd.concat([self._player_weeks(y) for y in all_seasons], ignore_index=True)
        games = self._read("schedules/games.csv")
        games = games[games["season"].isin(all_seasons) & (games["game_type"] == "REG")]
        teams = pd.concat([self._team_weeks(y) for y in all_seasons], ignore_index=True)
        teams = self._attach_scores(teams, games)
        weeks = self._attach_team_shares(weeks, teams)
        births = self._birth_dates()

        self.player_weeks = weeks
        self.player_stats = self._season_totals(weeks, births)
        self.team_stats = teams
        self.schedule = games[games["season"] == self.current_season][
            ["game_id", "season", "week", "gameday", "away_team", "home_team",
             "away_score", "home_score"]].reset_index(drop=True)
        self.current_rosters = self._current_rosters(births)

        os.makedirs(self.cache_dir, exist_ok=True)
        self.player_weeks.to_csv(self._path(CACHE_FILES["player_weeks"]), index=False)
        self.player_stats.to_csv(self._path(CACHE_FILES["player_stats"]), index=False)
        self.team_stats.to_csv(self._path(CACHE_FILES["team_games"]), index=False)
        self.schedule.to_csv(self._path(CACHE_FILES["schedule"]), index=False)
        self.current_rosters.to_csv(self._path(CACHE_FILES["current_rosters"]), index=False)

    def get_active_players(self) -> pd.DataFrame:
        """Skill players on an active 2026 roster, with their current team."""
        r = self.current_rosters
        return r[(r["status"] == "ACT") & r["position"].isin(SKILL_POSITIONS)].reset_index(drop=True)

    @property
    def last_completed_week(self) -> int:
        played = self.schedule.dropna(subset=["home_score"])
        return int(played["week"].max()) if len(played) else 0

    # ----------------------------------------------------------------- helpers
    def _path(self, name: str) -> str:
        return os.path.join(self.cache_dir, name)

    def _read(self, rel: str) -> pd.DataFrame:
        if self.source.startswith("http"):
            return pd.read_csv(f"{self.source}/{rel}", low_memory=False)
        return pd.read_csv(os.path.join(self.source, os.path.basename(rel)), low_memory=False)

    def _player_weeks(self, season: int) -> pd.DataFrame:
        raw = self._read(f"stats_player/stats_player_week_{season}.csv")
        raw = raw[(raw["season_type"] == "REG") & raw["position"].isin(SKILL_POSITIONS)].copy()
        raw["fumbles_lost"] = (_num(raw, "sack_fumbles_lost") + _num(raw, "rushing_fumbles_lost")
                               + _num(raw, "receiving_fumbles_lost"))
        raw["two_pt"] = (_num(raw, "passing_2pt_conversions") + _num(raw, "rushing_2pt_conversions")
                         + _num(raw, "receiving_2pt_conversions"))
        raw["nflverse_ppr"] = _num(raw, "fantasy_points_ppr")
        out = raw[["player_id", "player_display_name", "position", "season", "week", "game_id",
                   "team", "opponent_team"]].rename(columns={"player_display_name": "name"})
        for col in PLAYER_STATS:
            out[col] = _num(raw, col)
        return out.reset_index(drop=True)

    def _team_weeks(self, season: int) -> pd.DataFrame:
        raw = self._read(f"stats_team/stats_team_week_{season}.csv")
        raw = raw[raw["season_type"] == "REG"].copy()
        raw["blocked_kicks"] = (_num(raw, "def_fg_blocks") + _num(raw, "def_punt_blocks")
                                + _num(raw, "def_pat_blocks"))
        out = raw[["season", "week", "game_id", "team", "opponent_team"]].copy()
        for col in TEAM_STATS:
            out[col] = _num(raw, col)
        return out.reset_index(drop=True)

    @staticmethod
    def _attach_scores(teams: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
        home = games[["game_id", "home_team", "home_score", "away_score"]].rename(
            columns={"home_team": "team", "home_score": "points_for", "away_score": "points_against"})
        away = games[["game_id", "away_team", "away_score", "home_score"]].rename(
            columns={"away_team": "team", "away_score": "points_for", "home_score": "points_against"})
        scores = pd.concat([home, away], ignore_index=True)
        return teams.merge(scores, on=["game_id", "team"], how="left")

    @staticmethod
    def _attach_team_shares(weeks: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        team_vol = teams[["game_id", "team", "targets", "carries"]].rename(
            columns={"targets": "team_targets", "carries": "team_carries"})
        return weeks.merge(team_vol, on=["game_id", "team"], how="left")

    def _birth_dates(self) -> pd.DataFrame:
        players = self._read("players/players.csv")
        return players[["gsis_id", "birth_date"]].dropna().rename(columns={"gsis_id": "player_id"})

    def _season_totals(self, weeks: pd.DataFrame, births: pd.DataFrame) -> pd.DataFrame:
        sums = PLAYER_STATS + ["team_targets", "team_carries"]
        g = weeks.sort_values("week").groupby(["player_id", "season"], as_index=False)
        totals = g[sums].sum()
        info = g.agg(name=("name", "last"), position=("position", "last"),
                     team=("team", "last"), games=("week", "nunique"))
        out = info.merge(totals, on=["player_id", "season"])
        out["target_share"] = (out["targets"] / out["team_targets"].where(out["team_targets"] != 0)).round(3)
        out["carry_share"] = (out["carries"] / out["team_carries"].where(out["team_carries"] != 0)).round(3)
        out = out.drop(columns=["team_targets", "team_carries"])
        out["partial"] = out["season"] == self.current_season
        return out.merge(births, on="player_id", how="left")

    def _current_rosters(self, births: pd.DataFrame) -> pd.DataFrame:
        r = self._read(f"rosters/roster_{self.current_season}.csv")
        r = r.sort_values("week").groupby("gsis_id", as_index=False).last()
        r = r[["gsis_id", "full_name", "team", "position", "status", "jersey_number",
               "years_exp", "birth_date", "week"]].rename(columns={"gsis_id": "player_id"})
        return r.reset_index(drop=True)
