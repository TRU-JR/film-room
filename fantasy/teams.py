"""Team-level helpers: offense/defense unit tables, D/ST scoring, matchup factors (R9)."""
from __future__ import annotations

import pandas as pd

from .scoring import DST_WEIGHTS, points_allowed_score

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

OFFENSE_STATS = {
    "points_for": "Points",
    "total_yards": "Total yards",
    "passing_yards": "Passing yards",
    "rushing_yards": "Rushing yards",
    "offensive_tds": "Offensive TDs",
    "turnovers": "Turnovers",
    "sacks_suffered": "Sacks taken",
    "skill_fpts": "Skill-player fantasy pts",
}
DEFENSE_STATS = {
    "points_against": "Points allowed",
    "yards_allowed": "Yards allowed",
    "pass_yards_allowed": "Pass yards allowed",
    "rush_yards_allowed": "Rush yards allowed",
    "def_sacks": "Sacks",
    "takeaways": "Takeaways",
    "def_tds": "Defensive TDs",
    "dst_fpts": "D/ST fantasy pts",
}
# Stats where a smaller number is better for the unit
LOWER_IS_BETTER = {"turnovers", "sacks_suffered", "points_against", "yards_allowed",
                   "pass_yards_allowed", "rush_yards_allowed"}


def build_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add opponent (allowed) stats and D/ST fantasy points to each team-game row."""
    tg = team_games.copy()
    tg["total_yards"] = tg["passing_yards"] + tg["rushing_yards"]
    tg["offensive_tds"] = tg["passing_tds"] + tg["rushing_tds"]
    tg["turnovers"] = tg["passing_interceptions"] + tg["fumbles_lost_total"]
    opp = tg[["game_id", "team", "passing_yards", "rushing_yards", "total_yards"]].rename(columns={
        "team": "opponent_team", "passing_yards": "pass_yards_allowed",
        "rushing_yards": "rush_yards_allowed", "total_yards": "yards_allowed"})
    tg = tg.merge(opp, on=["game_id", "opponent_team"], how="left")
    tg["takeaways"] = tg["def_interceptions"] + tg["fumble_recovery_opp"]
    tg["dst_fpts"] = dst_points(tg)
    return tg


def dst_points(tg: pd.DataFrame) -> pd.Series:
    pts = pd.Series(0.0, index=tg.index)
    for col, w in DST_WEIGHTS.items():
        if col in tg:
            pts += tg[col].fillna(0) * w
    pa = tg["points_against"].apply(lambda v: points_allowed_score(v) if pd.notna(v) else 0.0)
    return (pts + pa).round(1)


def skill_fpts_by_team_game(player_weeks: pd.DataFrame) -> pd.DataFrame:
    """Fantasy points scored by a team's skill players, and allowed by each defense, per game."""
    return player_weeks.groupby(["game_id", "team"], as_index=False)["fpts"].sum().rename(
        columns={"fpts": "skill_fpts"})


def team_season_table(tg: pd.DataFrame, season: int, unit: str = "Offense",
                      player_weeks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-game averages for every team in one season, for the offense or defense unit."""
    rows = tg[tg["season"] == season].copy()
    if player_weeks is not None and "fpts" in player_weeks:
        rows = rows.merge(skill_fpts_by_team_game(player_weeks), on=["game_id", "team"], how="left")
    stats = OFFENSE_STATS if unit == "Offense" else DEFENSE_STATS
    cols = [c for c in stats if c in rows]
    out = rows.groupby("team")[cols].mean().round(1)
    out.insert(0, "games", rows.groupby("team").size())
    return out.reset_index()


def fpts_allowed_by_position(player_weeks: pd.DataFrame, season: int) -> pd.DataFrame:
    """Average fantasy points each defense allowed per game to QB/RB/WR/TE."""
    pw = player_weeks[player_weeks["season"] == season]
    per_game = pw.groupby(["opponent_team", "game_id", "position"])["fpts"].sum().reset_index()
    out = per_game.groupby(["opponent_team", "position"])["fpts"].mean().unstack().round(1)
    out.index.name = "team"
    return out.reset_index()


def matchup_factors(player_weeks: pd.DataFrame, current_season: int,
                    clip: tuple[float, float] = (0.8, 1.2)) -> pd.DataFrame:
    """Defense-vs-position multipliers relative to league average.

    Uses last season and this season so far, with this season's games counted twice.
    A factor of 1.10 means that defense allows 10% more points than average to the position.
    """
    pw = player_weeks[player_weeks["season"].isin([current_season - 1, current_season])].copy()
    pw["w"] = (pw["season"] == current_season).map({True: 2.0, False: 1.0})
    per_game = pw.groupby(["opponent_team", "game_id", "position"]).agg(
        fpts=("fpts", "sum"), w=("w", "first")).reset_index()
    per_game["wf"] = per_game["fpts"] * per_game["w"]
    agg = per_game.groupby(["opponent_team", "position"]).agg(wf=("wf", "sum"), w=("w", "sum"))
    allowed = (agg["wf"] / agg["w"]).unstack()
    factors = (allowed / allowed.mean()).clip(*clip).round(3)
    factors.index.name = "team"
    return factors


def dst_season_rows(tg: pd.DataFrame, current_season: int) -> pd.DataFrame:
    """D/ST 'player' seasons so defenses can be projected and rostered like players."""
    g = tg.groupby(["team", "season"], as_index=False).agg(
        games=("game_id", "nunique"), fpts=("dst_fpts", "sum"),
        def_sacks=("def_sacks", "sum"), takeaways=("takeaways", "sum"),
        points_against=("points_against", "mean"))
    g["fpts_pg"] = (g["fpts"] / g["games"]).round(2)
    g["player_id"] = "DST-" + g["team"]
    g["name"] = g["team"].map(TEAM_NAMES).fillna(g["team"]) + " D/ST"
    g["position"] = "DST"
    g["partial"] = g["season"] == current_season
    return g


def team_offense_points_factor(tg: pd.DataFrame, current_season: int,
                               clip: tuple[float, float] = (0.8, 1.2)) -> pd.Series:
    """How kind an offense is to opposing D/STs (D/ST points allowed vs league average)."""
    rows = tg[tg["season"].isin([current_season - 1, current_season])]
    by_opp = rows.groupby("opponent_team")["dst_fpts"].mean()
    return (by_opp / by_opp.mean()).clip(*clip).round(3)
