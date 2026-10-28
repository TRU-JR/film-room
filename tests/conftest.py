import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fantasy import Player, ScoringFormat, StatProcessor  # noqa: E402


def season_row(pid, name, pos, season, games, **stats):
    row = {"player_id": pid, "name": name, "position": pos, "team": "KC", "season": season,
           "games": games, "birth_date": "2000-01-15", "partial": False}
    row.update(stats)
    return row


@pytest.fixture
def seasons() -> pd.DataFrame:
    """Small synthetic player-season table, already processed with PPR scoring."""
    rows = [
        # a receiver who improves every year
        season_row("WR1", "Rising Receiver", "WR", 2022, 17, targets=80, receptions=50, receiving_yards=600, receiving_tds=4),
        season_row("WR1", "Rising Receiver", "WR", 2023, 17, targets=110, receptions=75, receiving_yards=900, receiving_tds=6),
        season_row("WR1", "Rising Receiver", "WR", 2024, 17, targets=140, receptions=100, receiving_yards=1200, receiving_tds=8),
        # a running back in decline
        season_row("RB1", "Fading Back", "RB", 2022, 17, carries=300, rushing_yards=1400, rushing_tds=12, receptions=40, receiving_yards=300),
        season_row("RB1", "Fading Back", "RB", 2023, 16, carries=250, rushing_yards=1000, rushing_tds=8, receptions=30, receiving_yards=200),
        season_row("RB1", "Fading Back", "RB", 2024, 15, carries=200, rushing_yards=700, rushing_tds=4, receptions=20, receiving_yards=150),
        # a rookie with a single season
        season_row("TE1", "Rookie End", "TE", 2024, 10, targets=50, receptions=40, receiving_yards=400, receiving_tds=2),
        # a quarterback
        season_row("QB1", "Steady Passer", "QB", 2023, 17, attempts=550, completions=370, passing_yards=4200, passing_tds=30,
                   passing_interceptions=10, carries=50, rushing_yards=250, rushing_tds=2),
        season_row("QB1", "Steady Passer", "QB", 2024, 17, attempts=560, completions=380, passing_yards=4300, passing_tds=31,
                   passing_interceptions=9, carries=55, rushing_yards=260, rushing_tds=2),
    ]
    df = pd.DataFrame(rows).fillna(0)
    return StatProcessor(df).run_all(ScoringFormat("PPR"))


@pytest.fixture
def players(seasons) -> dict:
    return {pid: Player.from_df(seasons, pid) for pid in seasons["player_id"].unique()}
