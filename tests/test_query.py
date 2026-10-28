import pytest

from fantasy import DataQuery, QueryError, ScoringFormat


def parsed(text):
    q = DataQuery()
    q.parse(text, last_completed=2024, current=2025)
    return q


def test_example_from_requirements():
    q = parsed("WRs under 25 with 100+ targets last season")
    assert q.filters["positions"] == ["WR"]
    assert q.filters["age"][1] < 25
    assert q.conditions == [("targets", ">=", 100.0)]
    assert q.filters["seasons"] == (2024, 2024)


@pytest.mark.parametrize("text, cond", [
    ("RB rushing yards > 1000", ("rushing_yards", ">", 1000.0)),
    ("RBs with over 1000 rushing yards", ("rushing_yards", ">", 1000.0)),
    ("QB passing tds at least 30", ("passing_tds", ">=", 30.0)),
    ("TEs with 8 receiving tds", ("receiving_tds", ">=", 8.0)),
    ("WR targets per game over 8", ("targets_pg", ">", 8.0)),
    ("WR target share above 25%", ("target_share", ">", 0.25)),
    ("RB fewer than 3 fumbles lost", ("fumbles_lost", "<", 3.0)),
])
def test_condition_forms(text, cond):
    assert parsed(text).conditions == [cond]


@pytest.mark.parametrize("text, seasons", [
    ("WR targets > 50 in 2022", (2022, 2022)),
    ("WR targets > 50 since 2022", (2022, 2024)),
    ("WR targets > 50 2021-2023", (2021, 2023)),
    ("WR targets > 50 last 2 seasons", (2023, 2024)),
    ("WR targets > 50 this season", (2025, 2025)),
])
def test_season_forms(text, seasons):
    assert parsed(text).filters["seasons"] == seasons


def test_by_season_team_and_limit():
    q = parsed("top 5 Chiefs WRs with 50+ receptions since 2022 by season")
    assert q.by_season and q.limit == 5
    assert q.filters["teams"] == ["KC"]


@pytest.mark.parametrize("text", ["", "   ", "show me something cool", "WR 100+ blocks"])
def test_bad_searches_raise(text):
    with pytest.raises(QueryError):
        parsed(text)


def test_run_single_season(seasons):
    q = parsed("WR receptions >= 75 in 2023")
    out = q.run(seasons, ScoringFormat("PPR"))
    assert list(out["player_id"]) == ["WR1"]
    assert q.chart_spec(out)["chart_type"] == "bar"


def test_run_combines_seasons_unless_by_season(seasons):
    q = parsed("WR receptions > 200 2022-2024")
    out = q.run(seasons, ScoringFormat("PPR"))
    assert out.iloc[0]["receptions"] == 225 and len(out) == 1
    q = parsed("WR receptions > 60 2022-2024 by season")
    out = q.run(seasons, ScoringFormat("PPR"))
    assert sorted(out["season"]) == [2022, 2023, 2024]
    assert q.chart_spec(out)["chart_type"] == "line"


def test_two_conditions_give_scatter(seasons):
    q = parsed("QBs with 4000+ passing yards and over 25 passing tds in 2024")
    out = q.run(seasons, ScoringFormat("PPR"))
    assert q.chart_spec(out) == {"chart_type": "scatter", "x": "passing_yards", "y": "passing_tds",
                                 "group_by": "position"}


def test_no_matches_raises(seasons):
    q = parsed("WR targets > 500 in 2024")
    with pytest.raises(QueryError, match="No players match"):
        q.run(seasons, ScoringFormat("PPR"))


def test_filters_and_summary(seasons):
    q = DataQuery(filters={"seasons": (2024, 2024), "positions": ["WR", "RB"], "min_games": 10},
                  group_by=["position"])
    out = q.apply(seasons)
    assert set(out["player_id"]) == {"WR1", "RB1"}
    summary = q.summary(out)
    assert list(summary["position"]) == ["RB", "WR"]
    assert list(summary["players"]) == [1, 1]
