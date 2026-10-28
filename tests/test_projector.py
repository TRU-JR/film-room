import math

import pandas as pd
import pytest

from fantasy import Player, Projector


def test_rising_player_is_flagged(players):
    proj = Projector(target_season=2025)
    p = players["WR1"]
    assert proj.slope(p) > 0
    assert proj.is_growing(p)
    assert proj.method(p) == "Regression"


def test_declining_player_is_not_flagged(players):
    proj = Projector(target_season=2025)
    assert proj.slope(players["RB1"]) < 0
    assert not proj.is_growing(players["RB1"])


def test_regression_extrapolates_one_season(players):
    # WR1 per-game points rise in a straight line, so 2025 continues the line
    p = players["WR1"]
    pg = p.seasons["fpts_pg"].tolist()
    step = pg[2] - pg[1]
    proj = Projector(target_season=2025)
    assert proj.project_per_game(p) == pytest.approx(pg[2] + step, abs=0.3)
    assert proj.project(p) == pytest.approx(proj.project_per_game(p) * 17, abs=0.1)


def test_rookie_projection_uses_optimism_factor(players):
    p = players["TE1"]
    proj = Projector(target_season=2025)
    expected = round(p.seasons["fpts_pg"].iloc[0] * 1.10, 2) * 17
    assert proj.project(p) == pytest.approx(expected, abs=0.1)
    assert proj.slope(p) == 0.0
    assert not proj.is_growing(p)


def test_no_games_means_no_projection():
    empty = Player("X", "Never Played", "WR", "KC")
    assert math.isnan(Projector().project(empty))


def test_projection_capped_by_ceiling():
    rows = pd.DataFrame({"season": [2023, 2024], "games": [17, 17], "fpts_pg": [2.0, 20.0],
                         "fpts": [34, 340], "partial": [False, False]})
    p = Player("J", "Jump", "WR", "KC", rows)
    assert Projector(target_season=2025).project_per_game(p) == pytest.approx(25.0)


def test_partial_season_ignored_by_regression(players, seasons):
    extra = seasons[seasons["player_id"] == "WR1"].iloc[[0]].copy()
    extra["season"], extra["partial"], extra["fpts_pg"] = 2025, True, 0.0
    p = Player("WR1", "Rising Receiver", "WR", "KC", pd.concat([players["WR1"].seasons, extra]))
    proj = Projector(target_season=2025)
    assert proj.project_per_game(p) == Projector(target_season=2025).project_per_game(players["WR1"])


def test_blended_moves_toward_current_pace(players):
    proj = Projector(target_season=2025)
    p = players["WR1"]
    base = proj.project_per_game(p)
    current = pd.Series({"games": 4, "fpts": 4 * (base + 8), "fpts_pg": base + 8})
    total = proj.blended(p, current, remaining_games=13)
    rate = 0.5 * (base + 8) + 0.5 * base
    assert total == pytest.approx(current["fpts"] + rate * 13, abs=0.2)


def test_backtest_reports_errors(seasons):
    bt = Projector().backtest(seasons, 2024, min_games=6)
    assert set(bt["player_id"]) == {"WR1", "RB1", "QB1"}   # TE1 has no earlier season
    row = bt.set_index("player_id").loc["RB1"]
    assert row["error_pg"] == pytest.approx(row["projected_pg"] - row["actual_pg"], abs=0.01)
