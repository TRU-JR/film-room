import os

import pandas as pd
import pytest

from fantasy import ScoringFormat, StatProcessor
from fantasy.scoring import points_allowed_score

LINE = pd.DataFrame([{"passing_yards": 300, "passing_tds": 2, "passing_interceptions": 1,
                      "rushing_yards": 50, "rushing_tds": 1, "receptions": 5,
                      "receiving_yards": 80, "receiving_tds": 1, "fumbles_lost": 1, "two_pt": 1}])


@pytest.mark.parametrize("fmt, expected", [
    # 12 + 8 - 2 + 5 + 6 + rec + 8 + 6 - 2 + 2 = 43 + rec
    ("PPR", 48.0), ("Half-PPR", 45.5), ("Standard", 43.0),
])
def test_scoring_formats(fmt, expected):
    assert ScoringFormat(fmt).points(LINE).iloc[0] == pytest.approx(expected)


def test_missing_columns_count_as_zero():
    df = pd.DataFrame([{"rushing_yards": 100}])
    assert ScoringFormat("PPR").points(df).iloc[0] == pytest.approx(10.0)


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        ScoringFormat("Superflex")


@pytest.mark.parametrize("allowed, pts", [(0, 10), (3, 7), (10, 4), (17, 1), (24, 0), (31, -1), (45, -4)])
def test_points_allowed_tiers(allowed, pts):
    assert points_allowed_score(allowed) == pts


def test_per_game_and_age(seasons):
    wr = seasons[(seasons["player_id"] == "WR1") & (seasons["season"] == 2024)].iloc[0]
    assert wr["fpts"] == pytest.approx(100 + 120 + 48)   # rec + yards/10 + TDs*6
    assert wr["fpts_pg"] == pytest.approx(round(268 / 17, 2))
    assert wr["age"] == pytest.approx(24.6, abs=0.1)     # born Jan 2000, measured Sept 1 2024
    assert wr["age_bin"] == "24–26"


CACHE = os.path.join(os.path.dirname(__file__), "..", "data_cache", "player_stats.csv")


@pytest.mark.skipif(not os.path.exists(CACHE), reason="nflverse cache not downloaded yet")
def test_ppr_matches_nflverse():
    raw = pd.read_csv(CACHE)
    scored = StatProcessor(raw).run_all(ScoringFormat("PPR"))
    assert (scored["fpts"] - scored["nflverse_ppr"]).abs().max() < 0.05
