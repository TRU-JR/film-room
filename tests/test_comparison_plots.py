import numpy as np
import pytest

from fantasy import Comparison, ComparisonError, Player, PlotBuilder, PlotError


def wr(i):
    return Player(f"w{i}", f"Receiver {i}", "WR", "KC")


def test_comparison_limits_to_six():
    c = Comparison()
    for i in range(6):
        c.add_player(wr(i))
    with pytest.raises(ComparisonError, match="up to 6"):
        c.add_player(wr(7))


def test_comparison_enforces_position_group(players):
    c = Comparison(stat="fpts")
    c.add_player(players["WR1"])
    with pytest.raises(ComparisonError, match="RB"):
        c.add_player(players["RB1"])
    with pytest.raises(ComparisonError, match="already"):
        c.add_player(players["WR1"])


def test_compare_players_table(players, seasons):
    c = Comparison(stat="fpts_pg")
    c.add_player(players["WR1"])
    c.add_player(Player.from_df(seasons.assign(player_id=seasons["player_id"].replace("TE1", "T")).assign(
        position="WR"), "T"))
    table = c.compare_players()
    assert list(table.index) == ["2022", "2023", "2024", "Career"]
    assert np.isnan(table.loc["2022", "Rookie End"])
    assert table.loc["Career", "Rising Receiver"] == pytest.approx(players["WR1"].seasons["fpts_pg"].mean(), abs=0.01)


def test_linear_trendline_perfect_fit():
    x = np.arange(10)
    fitted, r2 = PlotBuilder.fit_trendline(x, 3 * x + 2, "linear")
    assert r2 == pytest.approx(1.0)
    assert fitted[4] == pytest.approx(14.0)


def test_polynomial_and_exponential_fits():
    x = np.linspace(1, 5, 20)
    _, r2 = PlotBuilder.fit_trendline(x, x ** 2 - x + 1, "polynomial")
    assert r2 == pytest.approx(1.0)
    fitted, r2 = PlotBuilder.fit_trendline(x, 2 * np.exp(0.5 * x), "exponential")
    assert r2 == pytest.approx(1.0)
    assert fitted[0] == pytest.approx(2 * np.exp(0.5))


def test_trendline_errors():
    with pytest.raises(PlotError):
        PlotBuilder.fit_trendline([1, 2], [3, 4], "linear")
    with pytest.raises(PlotError):
        PlotBuilder.fit_trendline([1, 2, 3], [3, 4, 5], "cubic")


@pytest.mark.parametrize("chart", ["scatter", "histogram", "box", "line", "bar"])
def test_every_chart_type_builds(seasons, chart):
    x = {"line": "season", "bar": "name"}.get(chart, "games")
    fig = PlotBuilder(chart, x, "fpts", "position", "linear" if chart == "scatter" else None).build(seasons)
    assert len(fig.data) >= 1


def test_plot_errors(seasons):
    with pytest.raises(PlotError):
        PlotBuilder("pie")
    with pytest.raises(PlotError):
        PlotBuilder("scatter", "not_a_column", "fpts").build(seasons)
    with pytest.raises(PlotError):
        PlotBuilder("scatter", "games", "fpts").build(seasons.iloc[0:0])
