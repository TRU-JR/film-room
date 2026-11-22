import pandas as pd

from fantasy.injuries import current_injuries, season_injuries


def rows(*items):
    cols = ["player_id", "season", "week", "team", "source", "status", "injury"]
    return pd.DataFrame(items, columns=cols)


def test_current_uses_each_teams_latest_report():
    df = rows(("a", 2026, 9, "KC", "report", "Out", "Knee"),        # old report: ignored
              ("b", 2026, 10, "KC", "report", "Questionable", "Ankle"),
              ("c", 2026, 9, "BUF", "report", "Doubtful", "Hamstring"))  # BUF on bye in week 10
    now = current_injuries(df, 2026)
    assert set(now) == {"b", "c"}
    assert now["b"].dot == "🟠" and not now["b"].likely_out
    assert now["c"].dot == "🔴" and now["c"].text == "Doubtful (hamstring)"


def test_reserve_list_beats_report_and_borrows_body_part():
    df = rows(("a", 2026, 6, "SF", "report", "Out", "Achilles"),
              ("a", 2026, 11, "SF", "reserve", "IR", None),
              ("z", 2026, 11, "SF", "report", "Questionable", "Toe"))
    now = current_injuries(df, 2026)
    assert now["a"].status == "IR" and now["a"].injury == "Achilles"


def test_other_seasons_are_not_current():
    df = rows(("a", 2025, 17, "SF", "reserve", "IR", None))
    assert current_injuries(df, 2026) == {}
    assert current_injuries(pd.DataFrame(), 2026) == {}


def test_season_injuries_counts_weeks_out_only():
    df = rows(("a", 2024, 1, "SF", "reserve", "IR", None),
              ("a", 2024, 2, "SF", "reserve", "IR", None),
              ("a", 2024, 3, "SF", "report", "Out", "Calf"),
              ("a", 2023, 5, "SF", "report", "Questionable", "Calf"))   # played: no notice
    out = season_injuries(df, "a")
    assert list(out["season"]) == [2024]
    assert out.iloc[0]["weeks_out"] == 3
    assert out.iloc[0]["note"] == "Injured (calf): out 3 weeks, on IR"
