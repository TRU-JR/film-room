"""Film Room: NFL fantasy analytics (Streamlit).

Run with:  streamlit run app.py
Pages: explore(), player_page(), compare(), teams(), projections(), roster(), search()
Session state: current roster, scoring format, filters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

import ui
from fantasy import (Comparison, DataLoader, DataQuery, FantasyError, PlotBuilder, Player,
                     Projector, Roster, ScoringFormat, StatProcessor)
from fantasy.player import build_players
from fantasy.plots import (CHART_TYPES, INK, POSITION_COLORS, SERIES, TRENDLINES, apply_theme)
from fantasy.query import GROUPABLE, STAT_LABELS, label
from fantasy.roster import FLEX_ELIGIBLE, STARTING_SLOTS
from fantasy.scouting import scouting_notes
from fantasy.teams import (DEFENSE_STATS, LOWER_IS_BETTER, OFFENSE_STATS, TEAM_NAMES,
                           build_team_games, dst_season_rows, fpts_allowed_by_position,
                           matchup_factors, team_offense_points_factor, team_season_table)

APP_DIR = Path(__file__).parent
CURRENT_SEASON = 2026
POSITIONS = ["QB", "RB", "WR", "TE"]
CHART_STATS = ["fpts", "fpts_pg", "age", "games", "targets", "targets_pg", "receptions",
               "receiving_yards", "receiving_tds", "carries", "carries_pg", "rushing_yards",
               "rushing_tds", "passing_yards", "passing_tds", "passing_interceptions", "attempts",
               "target_share", "carry_share", "yards_per_carry", "yards_per_target", "catch_rate",
               "opportunities_pg", "scrimmage_yards", "total_tds", "season"]
BLUE_SCALE = LinearSegmentedColormap.from_list("fr_blue", ["#ffffff", "#9ec5f4", "#3987e5"])

st.set_page_config(page_title="Film Room", page_icon="🏈", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()


# =============================================================== data layer
@st.cache_resource(show_spinner="Loading NFL data…")
def get_loader() -> DataLoader:
    loader = DataLoader(cache_dir=str(APP_DIR / "data_cache"), current_season=CURRENT_SEASON)
    loader.load()
    return loader


@dataclass
class World:
    scoring: ScoringFormat
    seasons: pd.DataFrame          # skill-player seasons, scored
    dst: pd.DataFrame              # D/ST seasons, scored
    weeks: pd.DataFrame            # skill-player weeks, scored
    team_games: pd.DataFrame
    players: dict[str, Player]     # skill players + D/ST, keyed by id
    table: pd.DataFrame            # projections for everyone
    backtest: pd.DataFrame
    factors: pd.DataFrame          # defense-vs-position multipliers
    offense_factor: pd.Series      # how kind each offense is to opposing D/STs
    last_week: int
    remaining: dict[str, int] = field(default_factory=dict)
    schedule: pd.DataFrame = field(default_factory=pd.DataFrame)


@st.cache_resource(show_spinner="Scoring every season and building projections…")
def build_world(scoring_name: str) -> World:
    loader = get_loader()
    scoring = ScoringFormat(scoring_name)
    active = loader.get_active_players()
    team_now = dict(zip(active["player_id"], active["team"]))

    seasons = StatProcessor(loader.player_stats).run_all(scoring)
    seasons["active"] = seasons["player_id"].isin(team_now)
    seasons["current_team"] = seasons["player_id"].map(team_now)
    weeks = loader.player_weeks.copy()
    weeks["fpts"] = scoring.points(weeks)
    tg = build_team_games(loader.team_stats)
    dst = dst_season_rows(tg, CURRENT_SEASON)

    players = build_players(seasons, team_now)
    players.update(build_players(dst))
    proj = Projector(target_season=CURRENT_SEASON)
    table = proj.projection_table(list(players.values()))

    # the season in progress: points so far + remaining games at a blended rate
    sched = loader.schedule
    last_week = loader.last_completed_week
    future = sched[sched["week"] > last_week]
    remaining = pd.concat([future["home_team"], future["away_team"]]).value_counts().to_dict()
    cur = pd.concat([seasons[seasons["season"] == CURRENT_SEASON][["player_id", "games", "fpts", "fpts_pg"]],
                     dst[dst["season"] == CURRENT_SEASON][["player_id", "games", "fpts", "fpts_pg"]]])
    cur = cur.rename(columns={"games": "cur_games", "fpts": "cur_fpts", "fpts_pg": "cur_pg"})
    table = table.merge(cur, on="player_id", how="left")
    table[["cur_games", "cur_fpts"]] = table[["cur_games", "cur_fpts"]].fillna(0)
    w = table["cur_games"] / (table["cur_games"] + 4.0)
    table["rate"] = np.where(table["cur_games"] > 0, w * table["cur_pg"].fillna(0) + (1 - w) * table["proj_pg"],
                             table["proj_pg"]).round(2)
    table["remaining"] = table["team"].map(remaining).fillna(0).astype(int)
    table["blended"] = (table["cur_fpts"] + table["rate"] * table["remaining"]).round(1)
    table["active"] = table["player_id"].isin(team_now) | (table["position"] == "DST")
    table["label"] = table["name"] + " (" + table["position"].replace("DST", "D/ST") + ", " + table["team"] + ")"

    backtest = proj.backtest(seasons, CURRENT_SEASON - 1)
    return World(scoring, seasons, dst, weeks, tg, players, table, backtest,
                 matchup_factors(weeks, CURRENT_SEASON), team_offense_points_factor(tg, CURRENT_SEASON),
                 last_week, remaining, sched)


def opponent(world: World, team: str, week: int) -> str | None:
    s = world.schedule[world.schedule["week"] == week]
    home = s[s["home_team"] == team]
    if len(home):
        return home.iloc[0]["away_team"]
    away = s[s["away_team"] == team]
    return away.iloc[0]["home_team"] if len(away) else None


def game_projection(world: World, pid: str, week: int) -> tuple[float, str]:
    """Single-game projection: blended per-game rate x opponent matchup factor."""
    row = world.table[world.table["player_id"] == pid]
    if row.empty:
        return (math.nan, "No projection")
    r = row.iloc[0]
    opp = opponent(world, r["team"], week)
    if opp is None:
        return (0.0, "Bye week")
    if r["position"] == "DST":
        factor = float(world.offense_factor.get(opp, 1.0))
    else:
        factor = float(world.factors.loc[opp, r["position"]]) if opp in world.factors.index else 1.0
    venue = "vs" if (world.schedule[(world.schedule["week"] == week)]["home_team"] == r["team"]).any() else "at"
    return (round(r["rate"] * factor, 1), f"{venue} {opp}, matchup x{factor:.2f}")


# ============================================================ shared state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("scoring", "PPR")
    ss.setdefault("roster", Roster("My team"))
    ss.setdefault("rosters", [ss["roster"]])
    ss.setdefault("opened_files", set())
    ss.setdefault("filters", {"seasons": (2021, 2025), "positions": POSITIONS, "min_games": 4,
                              "age": (20, 40), "active_only": False})
    ss.setdefault("player_id", None)
    ss.setdefault("search_text", "WRs under 25 with 100+ targets last season")
    ss.setdefault("search_box", ss["search_text"])


def sidebar(world: World) -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown("## Film Room")
        ui.note(f"nflverse data, {world.seasons['season'].min()}–{CURRENT_SEASON}. "
                f"{CURRENT_SEASON} is complete through week {world.last_week}.")
        choice = st.segmented_control("Scoring format", ScoringFormat.all_formats(),
                                      default=ss["scoring"], key="scoring_ctl")
        if choice and choice != ss["scoring"]:
            ss["scoring"] = choice
            ss["roster"].scoring = ScoringFormat(choice)
            st.rerun()
        st.divider()
        st.markdown("**Filters**")
        f = ss["filters"]
        f["seasons"] = st.slider("Seasons", 2021, CURRENT_SEASON, f["seasons"],
                                 help=f"{CURRENT_SEASON} only has games through week {world.last_week}.")
        f["positions"] = st.multiselect("Positions", POSITIONS, f["positions"]) or POSITIONS
        f["min_games"] = st.slider("Minimum games", 0, 17, f["min_games"])
        f["age"] = st.slider("Age range", 20, 40, f["age"])
        f["active_only"] = st.toggle("Active players only", f["active_only"],
                                     help=f"Only players on a {CURRENT_SEASON} active roster.")
        st.divider()
        if st.button("Download latest data", help="Re-downloads every file from nflverse (about 30 seconds)."):
            try:
                with st.spinner("Downloading from nflverse…"):
                    get_loader().refresh()
                st.cache_resource.clear()
                st.rerun()
            except Exception as exc:  # network / format problems
                st.error(f"The download failed, so the app is still using the cached data. Details: {exc}")


def filtered(world: World) -> pd.DataFrame:
    f = st.session_state["filters"]
    q = DataQuery(filters={"seasons": f["seasons"], "positions": f["positions"],
                           "min_games": f["min_games"], "age": f["age"]})
    df = q.apply(world.seasons)
    if f["active_only"]:
        df = df[df["active"]]
    return df


def show_table(df: pd.DataFrame, cols: list[str], height: int | None = None, key: str | None = None,
               select: bool = False, rename: dict | None = None):
    cols = [c for c in cols if c in df]
    view = df[cols].copy()
    names = {c: (rename or {}).get(c, STAT_LABELS.get(c, label(c))) for c in cols}
    names.update({"name": "Player", "position": "Pos", "team": "Team", "current_team": "Team"})
    config = {}
    for c in cols:
        if c in ("season", "games", "players", "seasons_used", "remaining", "cur_games"):
            config[names[c]] = st.column_config.NumberColumn(format="%d")
        elif c.endswith("_share") or c == "catch_rate":
            config[names[c]] = st.column_config.NumberColumn(format="percent")
        elif pd.api.types.is_float_dtype(view[c]):
            config[names[c]] = st.column_config.NumberColumn(format="%.1f")
    view = view.rename(columns=names)
    kwargs = dict(hide_index=True, width="stretch", column_config=config, key=key)
    if height:
        kwargs["height"] = height
    if select:
        kwargs.update(on_select="rerun", selection_mode="single-row")
    return st.dataframe(view, **kwargs)


def fmt(v: float, digits: int = 1) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{digits}f}"


def open_player(pid: str) -> None:
    st.session_state["player_id"] = pid
    st.switch_page(PAGES["player"])


# ================================================================== pages
def explore() -> None:
    world = build_world(st.session_state["scoring"])
    df = filtered(world)
    f = st.session_state["filters"]
    ui.hero("Find the players who score",
            "Filter every player-season, chart any two stats against each other, and see averages "
            "for the groups you care about.", kicker=f"{world.scoring.name} scoring")
    lo, hi = f["seasons"]
    ui.board([
        (f"{len(df):,}", "Player-seasons in view", f"{lo}–{hi}" if lo != hi else str(lo)),
        (f"{df['player_id'].nunique():,}", "Players", ", ".join(f["positions"])),
        (fmt(df["fpts_pg"].mean()), "Average points per game", f"{f['min_games']}+ games"),
        (fmt(df["age"].mean()), "Average age", f"{f['age'][0]}–{f['age'][1]}"),
    ])
    if df.empty:
        st.info("No player-seasons match these filters. Widen the season range or lower the minimum games in the sidebar.")
        return

    with st.container(border=True, key="card_1"):
        st.markdown("### Build a chart")
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.2, 1.2, 1, 1.2])
        chart = c1.selectbox("Chart", list(k for k in CHART_TYPES if k != "bar"), format_func=CHART_TYPES.get)
        x_opts = CHART_STATS
        x = c2.selectbox("X axis" if chart != "box" else "Group axis (uses Group by)", x_opts,
                         index=x_opts.index("opportunities_pg") if chart != "line" else x_opts.index("season"),
                         format_func=label, disabled=chart == "box")
        y = c3.selectbox("Y axis", CHART_STATS, index=CHART_STATS.index("fpts_pg"), format_func=label,
                         disabled=chart == "histogram")
        group = c4.selectbox("Group by", [None] + list(GROUPABLE),
                             index=1, format_func=lambda g: "Nothing" if g is None else GROUPABLE[g])
        trend = c5.selectbox("Trendline", list(TRENDLINES), format_func=TRENDLINES.get,
                             index=1, disabled=chart != "scatter")
        try:
            builder = PlotBuilder(chart, x, y, group, trend if chart == "scatter" else None)
            st.plotly_chart(builder.build(df), width="stretch", config={"displaylogo": False})
            if builder.r2 is not None:
                strength = "strong" if builder.r2 >= 0.6 else ("moderate" if builder.r2 >= 0.3 else "weak")
                ui.note(f"{TRENDLINES[trend]} fit across all points: R² = {builder.r2:.2f} ({strength}). "
                        "Hover any point to see the player and season.")
        except FantasyError as e:
            st.warning(str(e))

    left, right = st.columns([1.05, 1])
    with left, st.container(border=True, key="card_2"):
        st.markdown("### Averages by group")
        groups = st.multiselect("Group by", list(GROUPABLE), default=["position"], format_func=GROUPABLE.get,
                                key="summary_groups")
        summary = DataQuery(group_by=groups).summary(df)
        show_table(summary, groups + ["players", "age", "games", "fpts", "fpts_pg", "targets",
                                      "carries", "total_tds"])
    with right, st.container(border=True, key="card_3"):
        st.markdown("### Top player-seasons")
        sort = st.selectbox("Rank by", CHART_STATS[:-1], format_func=label, key="top_sort")
        top = df.sort_values(sort, ascending=False).head(60)
        ev = show_table(top, ["name", "position", "season", "team", "age", "games", sort, "fpts_pg"],
                        height=360, key="top_table", select=True)
        ui.note("Select a row to open that player's page.")
        if ev and ev.selection.rows:
            open_player(top.iloc[ev.selection.rows[0]]["player_id"])


def _player_options(world: World) -> list[str]:
    t = world.table[world.table["position"] != "DST"].copy()
    t["sort"] = t["active"].astype(int) * 1000 + t["projection"]
    ids = list(t.sort_values("sort", ascending=False)["player_id"])
    others = [p for p in world.seasons.sort_values("fpts", ascending=False)["player_id"].unique() if p not in set(ids)]
    return ids + others


def player_page() -> None:
    world = build_world(st.session_state["scoring"])
    options = _player_options(world)
    current = st.session_state.get("player_id")
    idx = options.index(current) if current in options else 0
    pid = st.selectbox("Player", options, index=idx, format_func=lambda i: world.players[i].label,
                       help="Type to search by name.")
    st.session_state["player_id"] = pid
    p = world.players[pid]
    proj = Projector(target_season=CURRENT_SEASON)
    row = world.table[world.table["player_id"] == pid]
    r = row.iloc[0] if len(row) else None

    age = p.seasons["age"].dropna().iloc[-1] if p.seasons["age"].notna().any() else math.nan
    active = bool(p.seasons["active"].iloc[-1])
    facts = [f"Age {age:.0f} in {int(p.seasons['season'].iloc[-1])}" if not math.isnan(age) else "Age unknown",
             f"{p.num_seasons()} season{'s' if p.num_seasons() != 1 else ''} since 2021",
             "Active" if active else f"Not on a {CURRENT_SEASON} active roster"]
    flags = []
    if r is not None and r["growing"]:
        flags.append((f"Growing: +{r['slope']:.1f} pts/game per season", "good"))
    elif r is not None and r["slope"] < -0.5:
        flags.append((f"Declining: {r['slope']:.1f} pts/game per season", "bad"))
    if r is not None:
        flags.append((f"Projection method: {r['method']}", ""))
    ui.player_header(p.name, p.position, TEAM_NAMES.get(p.team, p.team), facts, flags)

    if r is not None:
        ui.board([
            (fmt(r["projection"], 0), f"{CURRENT_SEASON} preseason projection", f"{fmt(r['proj_pg'])} per game x 17"),
            (f"{fmt(r['low'], 0)}–{fmt(r['high'], 0)}", "Likely range", "plus or minus one season of variation"),
            (fmt(r["cur_pg"]) if r["cur_games"] else "–", f"{CURRENT_SEASON} points per game",
             f"{int(r['cur_games'])} games played" if r["cur_games"] else "No games yet"),
            (fmt(r["blended"], 0), f"{CURRENT_SEASON} full-season outlook",
             f"actual + {int(r['remaining'])} games at {fmt(r['rate'])}"),
        ])
    left, right = st.columns([1.5, 1])
    with left, st.container(border=True, key="card_4"):
        st.markdown("### Points per game by season")
        st.plotly_chart(_trend_figure(p, proj, r), width="stretch", config={"displaylogo": False})
    with right, st.container(border=True, key="card_5"):
        st.markdown("### Scouting notes")
        notes = scouting_notes(p, world.seasons, proj, CURRENT_SEASON, world.scoring.name)
        st.markdown('<ul class="fr-notes">' + "".join(f"<li>{n}</li>" for n in notes) + "</ul>",
                    unsafe_allow_html=True)
        ui.note("Written from the stats by fixed rules, not by an AI model.")
        if st.button(f"Add {p.name.split()[-1]} to {st.session_state['roster'].name}", type="primary"):
            try:
                slot = st.session_state["roster"].add(p)
                st.success(f"Added to {slot}.")
            except FantasyError as e:
                st.error(str(e))

    tabs = st.tabs(["Season totals", "Per game", f"{CURRENT_SEASON} game log", "Career"])
    base = ["season", "team", "age", "games", "fpts", "fpts_pg"]
    by_pos = {"QB": ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
                     "carries", "rushing_yards", "rushing_tds"],
              "RB": ["carries", "rushing_yards", "rushing_tds", "yards_per_carry", "targets", "receptions",
                     "receiving_yards", "receiving_tds", "carry_share"],
              "WR": ["targets", "receptions", "receiving_yards", "receiving_tds", "yards_per_target",
                     "target_share", "catch_rate", "carries", "rushing_yards"],
              "TE": ["targets", "receptions", "receiving_yards", "receiving_tds", "yards_per_target",
                     "target_share", "catch_rate"]}[p.position]
    with tabs[0]:
        show_table(p.seasons, base + by_pos)
    with tabs[1]:
        show_table(p.seasons, ["season", "games"] + [c for c in p.seasons.columns if c.endswith("_pg")])
    with tabs[2]:
        log = world.weeks[(world.weeks["player_id"] == pid) & (world.weeks["season"] == CURRENT_SEASON)]
        if log.empty:
            st.info(f"No {CURRENT_SEASON} games yet.")
        else:
            show_table(log.sort_values("week"), ["week", "team", "opponent_team", "fpts"] + [
                c for c in by_pos if c in log], rename={"week": "Week", "opponent_team": "Opponent"})
    with tabs[3]:
        totals = p.career_totals()
        show_table(totals.to_frame().T, [c for c in ["games", "fpts", "fpts_pg"] + by_pos if c in totals])


def _trend_figure(p: Player, proj: Projector, r) -> go.Figure:
    s = p.seasons
    done = s[~s["partial"].astype(bool)]
    cur = s[s["partial"].astype(bool)]
    color = POSITION_COLORS.get(p.position, SERIES[0])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=done["season"], y=done["fpts_pg"], name="Completed season", marker_color=color,
                         customdata=done["games"],
                         hovertemplate="%{x}: %{y:.1f} pts/game over %{customdata} games<extra></extra>"))
    if len(cur):
        fig.add_trace(go.Bar(x=cur["season"], y=cur["fpts_pg"], name=f"{CURRENT_SEASON} so far",
                             marker=dict(color="rgba(0,0,0,0)", line=dict(color=color, width=2),
                                         pattern=dict(shape="/", fgcolor=color, size=7)),
                             customdata=cur["games"],
                             hovertemplate="%{x} so far: %{y:.1f} pts/game over %{customdata} games<extra></extra>"))
    hist = p.completed_seasons(before=CURRENT_SEASON)
    if len(hist) >= 2:
        fitted, _ = PlotBuilder.fit_trendline(hist["season"], hist["fpts_pg"], "linear")
        xs = list(hist["season"]) + [CURRENT_SEASON]
        slope = (fitted[-1] - fitted[0]) / max(hist["season"].iloc[-1] - hist["season"].iloc[0], 1)
        ys = list(fitted) + [fitted[-1] + slope * (CURRENT_SEASON - hist["season"].iloc[-1])]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Trend", line=dict(color=INK, width=2, dash="dash"),
                                 hoverinfo="skip"))
    if r is not None and not math.isnan(r["proj_pg"]):
        lo, hi = r["low"] / 17, r["high"] / 17
        fig.add_trace(go.Scatter(x=[CURRENT_SEASON], y=[r["proj_pg"]], mode="markers", name="Projection",
                                 marker=dict(symbol="diamond", size=14, color=INK, line=dict(width=2, color="#fff")),
                                 error_y=dict(type="data", symmetric=False, array=[hi - r["proj_pg"]],
                                              arrayminus=[r["proj_pg"] - lo], color=INK, thickness=1.5, width=8),
                                 hovertemplate=f"Projected {CURRENT_SEASON}: %{{y:.1f}} pts/game<br>Range {lo:.1f}–{hi:.1f}<extra></extra>"))
    fig.update_layout(barmode="overlay", xaxis_title="", yaxis_title="Fantasy points per game")
    fig.update_xaxes(dtick=1)
    return apply_theme(fig, height=380)


def compare() -> None:
    world = build_world(st.session_state["scoring"])
    ui.hero("Line them up", "Put up to six players from one position side by side on any stat, season by season.")
    c1, c2, c3 = st.columns([0.8, 2.6, 1.2])
    pos = c1.selectbox("Position", POSITIONS, index=2)
    pool = world.table[(world.table["position"] == pos) & world.table["active"]].sort_values("projection", ascending=False)
    default = list(pool["player_id"].head(3))
    picked = c2.multiselect("Players (up to six)", list(pool["player_id"]), default=default, max_selections=6,
                            format_func=lambda i: world.players[i].label, key=f"cmp_{pos}")
    stats = ["fpts", "fpts_pg", "games", "targets", "targets_pg", "receptions", "receiving_yards", "receiving_tds",
             "carries", "rushing_yards", "rushing_tds", "passing_yards", "passing_tds", "target_share",
             "yards_per_target", "yards_per_carry", "opportunities_pg"]
    stat = c3.selectbox("Stat", stats, index=1, format_func=label)
    comp = Comparison(stat=stat)
    try:
        for pid in picked:
            comp.add_player(world.players[pid])
        table = comp.compare_players()
    except FantasyError as e:
        st.error(str(e))
        return
    if table.empty:
        st.info("Pick at least two players to compare.")
        return
    with st.container(border=True, key="card_6"):
        st.markdown(f"### {label(stat)} by season")
        fig = go.Figure()
        by_season = table.drop(index="Career")
        for i, name in enumerate(by_season.columns):
            col = by_season[name].dropna()
            fig.add_trace(go.Scatter(x=col.index, y=col.values, mode="lines+markers", name=name,
                                     line=dict(width=2.5, color=SERIES[i]),
                                     marker=dict(size=9, color=SERIES[i], line=dict(width=2, color="#fff")),
                                     hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:,.1f}}<extra></extra>"))
        fig.update_layout(yaxis_title=label(stat), hovermode="x unified")
        st.plotly_chart(apply_theme(fig, height=420), width="stretch", config={"displaylogo": False})
        if CURRENT_SEASON in world.seasons["season"].values:
            ui.note(f"{CURRENT_SEASON} is in progress (through week {world.last_week}); totals are partial.")
    with st.container(border=True, key="card_7"):
        st.markdown("### Side by side")
        shown = table.reset_index().rename(columns={"index": "Season", "season": "Season"})
        st.dataframe(shown, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f" if "share" in stat else "%.1f")
                                    for c in shown.columns if c != "Season"})
        proj_rows = world.table[world.table["player_id"].isin(picked)][
            ["name", "proj_pg", "projection", "slope", "growing", "cur_pg", "blended"]]
        st.markdown(f"**{CURRENT_SEASON} outlook**")
        show_table(proj_rows, ["name", "proj_pg", "projection", "slope", "cur_pg", "blended"],
                   rename={"proj_pg": "Proj. pts/game", "projection": "Preseason proj.", "slope": "Trend (pts/game/yr)",
                           "cur_pg": f"{CURRENT_SEASON} pts/game", "blended": "Full-season outlook"})


def teams() -> None:
    world = build_world(st.session_state["scoring"])
    ui.hero("Offenses and defenses", "Rank every unit, put two teams head to head, and find the defenses that "
            "give up fantasy points, the matchups to target.")
    seasons = sorted(world.team_games["season"].unique(), reverse=True)
    tab1, tab2, tab3 = st.tabs(["Unit rankings", "Head to head", "Fantasy points allowed"])
    with tab1:
        c1, c2, c3 = st.columns([1, 1, 2])
        season = c1.selectbox("Season", seasons, index=1 if len(seasons) > 1 else 0, key="t_season")
        unit = c2.segmented_control("Unit", ["Offense", "Defense"], default="Offense", key="t_unit") or "Offense"
        stats = OFFENSE_STATS if unit == "Offense" else DEFENSE_STATS
        stat = c3.selectbox("Rank by", list(stats), format_func=stats.get, key="t_stat",
                            index=list(stats).index("dst_fpts") if unit == "Defense" else 0)
        comp = Comparison(unit=unit)
        tbl = comp.compare_teams(world.team_games, season, world.weeks)
        asc = stat in LOWER_IS_BETTER
        tbl = tbl.sort_values(stat, ascending=asc).reset_index(drop=True)
        with st.container(border=True, key="card_8"):
            st.markdown(f"### {stats[stat]} per game, {season}")
            ui.note(("Lower is better" if asc else "Higher is better") + " for this stat; best unit at the top.")
            fig = go.Figure(go.Bar(y=tbl["team"], x=tbl[stat], orientation="h", marker_color=SERIES[0],
                                   hovertemplate="<b>%{y}</b>: %{x:.1f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
            fig.update_layout(xaxis_title=f"{stats[stat]} per game", bargap=0.18)
            st.plotly_chart(apply_theme(fig, height=720, legend=False), width="stretch", config={"displaylogo": False})
        with st.container(border=True, key="card_9"):
            st.markdown("### Full table")
            view = tbl.rename(columns={**stats, "team": "Team", "games": "Games"})
            styler = view.style.format(precision=1)
            for col, name in stats.items():
                if name in view:
                    styler = styler.background_gradient(cmap=BLUE_SCALE if col not in LOWER_IS_BETTER else BLUE_SCALE.reversed(),
                                                        subset=[name])
            st.dataframe(styler, hide_index=True, width="stretch", height=560)
    with tab2:
        c1, c2, c3 = st.columns(3)
        team_list = sorted(TEAM_NAMES)
        a = c1.selectbox("Team", team_list, index=team_list.index("KC"), format_func=lambda t: TEAM_NAMES[t])
        b = c2.selectbox("Opponent", team_list, index=team_list.index("BUF"), format_func=lambda t: TEAM_NAMES[t])
        season = c3.selectbox("Season", seasons, key="h2h_season")
        if a == b:
            st.warning("Pick two different teams.")
        else:
            for unit, stats in (("Offense", OFFENSE_STATS), ("Defense", DEFENSE_STATS)):
                tbl = team_season_table(world.team_games, season, unit, world.weeks).set_index("team")
                rows = []
                for col, name in stats.items():
                    ranks = tbl[col].rank(ascending=col in LOWER_IS_BETTER, method="min")
                    va, vb = tbl.loc[a, col], tbl.loc[b, col]
                    better = a if ((va < vb) if col in LOWER_IS_BETTER else (va > vb)) else b
                    rows.append({"Stat (per game)": name, a: va, f"{a} rank": int(ranks[a]),
                                 b: vb, f"{b} rank": int(ranks[b]), "Edge": better if va != vb else "Even"})
                with st.container(border=True, key=f"card_10_{unit}"):
                    st.markdown(f"### {unit}")
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                                 column_config={a: st.column_config.NumberColumn(format="%.1f"),
                                                b: st.column_config.NumberColumn(format="%.1f")})
            ui.note("Ranks are out of 32 teams; 1 is best for that stat.")
    with tab3:
        season = st.selectbox("Season", seasons, key="fpa_season")
        allowed = fpts_allowed_by_position(world.weeks, season)
        dst = team_season_table(world.team_games, season, "Defense")[["team", "dst_fpts"]]
        allowed = allowed.merge(dst, on="team").sort_values("WR", ascending=False)
        allowed = allowed[["team", "QB", "RB", "WR", "TE", "dst_fpts"]].rename(
            columns={"team": "Defense", "dst_fpts": "Own D/ST pts"})
        with st.container(border=True, key="card_11"):
            st.markdown(f"### {world.scoring.name} points allowed per game, {season}")
            ui.note("Darker cells give up more points to that position, so they're matchups to target. "
                    "The last column is the defense's own D/ST scoring.")
            styler = allowed.style.format(precision=1).background_gradient(cmap=BLUE_SCALE, subset=["QB", "RB", "WR", "TE"]) \
                .background_gradient(cmap=LinearSegmentedColormap.from_list("g", ["#ffffff", "#9ad49a"]), subset=["Own D/ST pts"])
            st.dataframe(styler, hide_index=True, width="stretch", height=760)


def projections() -> None:
    world = build_world(st.session_state["scoring"])
    ui.hero(f"{CURRENT_SEASON} projections",
            "Linear regression on each player's per-game scoring since 2021, then blended with what's happened "
            f"so far this season. Players with a rising trend are flagged.", kicker=f"{world.scoring.name} scoring")
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    pos = c1.segmented_control("Position", POSITIONS + ["D/ST"], default="WR", key="proj_pos") or "WR"
    growing = c2.toggle("Growing players only", False)
    active = c3.toggle("Active only", True)
    sort = c4.selectbox("Sort by", ["blended", "projection", "slope", "cur_pg"],
                        format_func={"blended": "Full-season outlook", "projection": "Preseason projection",
                                     "slope": "Trend", "cur_pg": f"{CURRENT_SEASON} pts/game"}.get)
    t = world.table[world.table["position"] == ("DST" if pos == "D/ST" else pos)]
    if growing:
        t = t[t["growing"]]
    if active:
        t = t[t["active"]]
    t = t.sort_values(sort, ascending=False).reset_index(drop=True)
    t["flag"] = np.where(t["growing"], "▲ Growing", np.where(t["slope"] < -0.5, "▼ Declining", ""))
    t["range"] = t["low"].round(0).astype(int).astype(str) + "–" + t["high"].round(0).astype(int).astype(str)
    with st.container(border=True, key="card_12"):
        ev = show_table(t, ["name", "team", "proj_pg", "projection", "range", "slope", "flag",
                            "cur_games", "cur_pg", "blended"], height=520, key="proj_table", select=pos != "D/ST",
                        rename={"proj_pg": "Proj. pts/game", "projection": "Preseason proj.", "range": "Likely range",
                                "slope": "Trend", "flag": "Flag", "cur_games": "Games",
                                "cur_pg": f"{CURRENT_SEASON} pts/game", "blended": "Full-season outlook",
                                "method": "Method"})
        ui.note("Trend is the change in points per game per season (a regression slope, so it needs 2+ seasons). One-season players are projected at their "
                "per-game rate x 1.10 over 17 games. Select a row to open the player.")
        if ev and pos != "D/ST" and ev.selection.rows:
            open_player(t.iloc[ev.selection.rows[0]]["player_id"])

    bt = world.backtest
    with st.container(border=True, key="card_13"):
        st.markdown(f"### How accurate is it? Backtest on {CURRENT_SEASON - 1}")
        ui.note(f"Each player was projected for {CURRENT_SEASON - 1} using only seasons before it, then compared with "
                f"what happened (players with 6+ games). Error is in fantasy points per game.")
        mae = bt["error_pg"].abs()
        ui.board([(fmt(mae.mean(), 2), "Average miss (pts/game)", f"{len(bt)} players"),
                  (f"{(mae <= 3).mean():.0%}", "Within 3 points per game", None),
                  (f"{bt['error_pg'].mean():+.2f}", "Average bias", "positive means projected too high")])
        left, right = st.columns([1, 1.4])
        with left:
            summary = bt.assign(abs_err=mae).groupby(["position", "method"]).agg(
                players=("player_id", "count"), mae=("abs_err", "mean"), bias=("error_pg", "mean")).round(2).reset_index()
            show_table(summary, ["position", "method", "players", "mae", "bias"],
                       rename={"mae": "Avg. miss", "bias": "Bias", "method": "Method"})
        with right:
            fig = go.Figure()
            for p in POSITIONS:
                part = bt[bt["position"] == p]
                fig.add_trace(go.Scatter(x=part["projected_pg"], y=part["actual_pg"], mode="markers", name=p,
                                         text=part["name"], marker=dict(size=8, color=POSITION_COLORS[p], opacity=.8,
                                                                        line=dict(width=1.5, color="#fff")),
                                         hovertemplate="<b>%{text}</b><br>Projected %{x:.1f}<br>Actual %{y:.1f}<extra></extra>"))
            top = float(max(bt["projected_pg"].max(), bt["actual_pg"].max())) + 1
            fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect projection",
                                     line=dict(color=INK, dash="dash", width=1.5), hoverinfo="skip"))
            fig.update_layout(xaxis_title="Projected pts/game", yaxis_title="Actual pts/game")
            st.plotly_chart(apply_theme(fig, height=400), width="stretch", config={"displaylogo": False})


def roster() -> None:
    world = build_world(st.session_state["scoring"])
    ss = st.session_state
    team: Roster = ss["roster"]
    ui.hero("Build your roster", "Fill a starting lineup, see its projected points for the season or a single week, "
            "and keep as many rosters as you like.", kicker=f"{world.scoring.name} scoring")

    left, right = st.columns([1, 1.6])
    with left:
        with st.container(border=True, key="card_14"):
            st.markdown("### Your rosters")
            books: list[Roster] = ss["rosters"]
            idx = next((i for i, r in enumerate(books) if r is team), 0)
            pick = st.selectbox("Working on", range(len(books)), index=idx,
                                format_func=lambda i: f"{books[i].name} ({len(books[i].players)} players)")
            if books[pick] is not team:
                ss["roster"] = books[pick]
                st.rerun()
            name = st.text_input("Roster name", team.name, key=f"rname_{id(team)}")
            if name.strip() and name != team.name:
                team.name = name.strip()
            b1, b2 = st.columns(2)
            if b1.button("New roster", width="stretch"):
                fresh = Roster(f"Roster {len(books) + 1}", scoring=world.scoring)
                books.append(fresh)
                ss["roster"] = fresh
                st.rerun()
            if b2.button("Delete roster", width="stretch", disabled=len(books) == 1):
                books.remove(team)
                ss["roster"] = books[0]
                st.rerun()
            st.download_button("Download roster file", team.to_json(), file_name=Roster.file_name(team.name),
                               mime="application/json", type="primary", width="stretch")
            uploads = st.file_uploader("Open roster files", type="json", accept_multiple_files=True,
                                       help="Pick one or more .json files you downloaded earlier.")
            opened_any = False
            for up in uploads or []:
                if up.file_id in ss["opened_files"]:
                    continue
                ss["opened_files"].add(up.file_id)
                try:
                    loaded = Roster.from_json(up.getvalue().decode("utf-8"), lookup=lambda pid: world.players[pid])
                    books.append(loaded)
                    ss["roster"] = loaded
                    opened_any = True
                except (FantasyError, UnicodeDecodeError) as e:
                    st.error(f"{up.name}: {e}")
            if opened_any:
                st.toast(f"Opened {ss['roster'].name}.")
                st.rerun()
            ui.note("Rosters stay in this browser tab while it's open. Download a roster to keep it, "
                    "then open the file next time to pick up where you left off.")

        with st.container(border=True, key="card_15"):
            st.markdown("### Add a player")
            pos = st.segmented_control("Position", ["QB", "RB", "WR", "TE", "D/ST"], default="RB", key="add_pos") or "RB"
            pool = world.table[(world.table["position"] == ("DST" if pos == "D/ST" else pos)) & world.table["active"]]
            pool = pool[~pool["player_id"].isin([p.player_id for p in team.players])].sort_values("blended", ascending=False)
            pid = st.selectbox("Player", list(pool["player_id"]), key="add_pid",
                               format_func=lambda i: f"{world.players[i].label}  {pool.set_index('player_id').loc[i, 'blended']:.0f} pts")
            if st.button("Add to lineup", type="primary", disabled=pid is None):
                try:
                    slot = team.add(world.players[pid])
                    st.toast(f"{world.players[pid].name} added at {slot}.")
                    st.rerun()
                except FantasyError as e:
                    st.error(str(e))

    with right:
        with st.container(border=True, key="card_16"):
            c1, c2 = st.columns([1.2, 1])
            view = c1.segmented_control("Projection", ["Season", "Single game"], default="Season", key="r_view") or "Season"
            weeks = sorted(world.schedule.loc[world.schedule["week"] > world.last_week, "week"].unique())
            week = c2.selectbox("Week", weeks, disabled=view == "Season") if weeks else None

            def points(p: Player) -> tuple[float, str]:
                if view == "Season":
                    row = world.table[world.table["player_id"] == p.player_id]
                    if row.empty:
                        return (math.nan, "No projection")
                    rr = row.iloc[0]
                    return (rr["blended"], f"{fmt(rr['cur_fpts'])} so far + {int(rr['remaining'])} games")
                return game_projection(world, p.player_id, week)

            html_rows = []
            for slot, p in team.lineup():
                if p is None:
                    html_rows.append(ui.slot_row(slot, None, "", "", bench=slot == "BENCH"))
                    continue
                pts, sub = points(p)
                html_rows.append(ui.slot_row(slot, p.name, f"{p.position.replace('DST', 'D/ST')}, {p.team}  |  {sub}",
                                             fmt(pts), bench=slot == "BENCH"))
            total = team.total_projection(lambda p: points(p)[0])
            starters = len(team.starters())
            open_starts = sum(team.open_slots(s) for s in STARTING_SLOTS)
            ui.board([(fmt(total), "Projected starter points",
                       f"{CURRENT_SEASON} season outlook" if view == "Season" else f"Week {week}"),
                      (str(starters), "Starters filled", f"{open_starts} open" if open_starts else "Lineup complete"),
                      (str(len(team.players) - starters), "On the bench", None)])
            st.markdown("".join(html_rows), unsafe_allow_html=True)
            ui.note("Season = points already scored this season + remaining games at a rate that blends the "
                    "projection with this season's pace. Single game = that rate x the opponent's matchup factor "
                    "(points it allows to the position vs league average).")

        if team.players:
            with st.container(border=True, key="card_17"):
                st.markdown("### Change the lineup")
                c1, c2, c3, c4 = st.columns([1.6, 1, 0.8, 0.8])
                who = c1.selectbox("Player", [p.player_id for p in team.players],
                                   format_func=lambda i: world.players[i].name if i in world.players else i)
                pl = next(p for p in team.players if p.player_id == who)
                slots = [s for s in STARTING_SLOTS + ["BENCH"]
                         if s in (pl.position.replace("DST", "D/ST"), "BENCH") or (s == "FLEX" and pl.position in FLEX_ELIGIBLE)]
                dest = c2.selectbox("Move to", slots)
                c3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                c4.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                if c3.button("Move", width="stretch"):
                    try:
                        team.move(pl, dest)
                        st.rerun()
                    except FantasyError as e:
                        st.error(str(e))
                if c4.button("Remove", width="stretch"):
                    team.remove(pl)
                    st.rerun()


EXAMPLES = ["WRs under 25 with 100+ targets last season",
            "RBs with 1000+ rushing yards since 2023 by season",
            "QBs with over 30 passing tds and 4000+ passing yards in 2024",
            "top 10 TEs with points per game over 10",
            "Lions WRs this season"]


def search() -> None:
    world = build_world(st.session_state["scoring"])
    ss = st.session_state
    ui.hero("Ask the data", "Type a question in plain English. The app reads positions, stat rules, ages, seasons "
            "and teams, then picks a chart for the answer.")
    def use_example():
        if ss.get("example"):
            ss["search_box"] = ss["example"]
    st.pills("Try one", EXAMPLES, key="example", on_change=use_example)
    with st.form("search_form", border=False):
        c1, c2 = st.columns([5, 1])
        text = c1.text_input("Question", key="search_box", label_visibility="collapsed",
                             placeholder="e.g. WRs under 25 with 100+ targets last season")
        c2.form_submit_button("Search", type="primary", width="stretch")
    ss["search_text"] = text
    q = DataQuery()
    try:
        q.parse(text, last_completed=CURRENT_SEASON - 1, current=CURRENT_SEASON)
        result = q.run(world.seasons, world.scoring)
    except FantasyError as e:
        st.error(str(e))
        with st.expander("What can I ask?"):
            st.markdown(SEARCH_HELP)
        return
    ui.chips(q.describe(), kind="gold")
    spec = q.chart_spec(result)
    plot_df = result
    if spec["chart_type"] == "line":
        top_ids = result.groupby("player_id")[q.stat].sum().nlargest(6).index
        plot_df = result[result["player_id"].isin(top_ids)]
    elif spec["chart_type"] == "bar":
        plot_df = result.drop_duplicates("player_id").head(15)
    with st.container(border=True, key="card_18"):
        n = result["player_id"].nunique()
        st.markdown(f"### {n} player{'s' if n != 1 else ''} match")
        try:
            fig = PlotBuilder(spec["chart_type"], spec["x"], spec["y"], spec["group_by"],
                              "linear" if spec["chart_type"] == "scatter" and len(plot_df) >= 3 else None).build(plot_df)
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
            if spec["chart_type"] == "line" and n > 6:
                ui.note("The chart shows the six highest totals; the table lists everyone.")
            elif spec["chart_type"] == "bar" and n > 15:
                ui.note("The chart shows the top 15; the table lists everyone.")
        except FantasyError as e:
            st.warning(str(e))
    with st.container(border=True, key="card_19"):
        stat_cols = list(dict.fromkeys(c for c, _, _ in q.conditions))
        seasons_col = ["season"] if "first_season" not in result else ["first_season", "season"]
        ev = show_table(result, ["name", "position", "team"] + seasons_col + ["age", "games"] + stat_cols + ["fpts", "fpts_pg"],
                        height=420, key="search_table", select=True,
                        rename={"first_season": "From", "season": "Season" if "first_season" not in result else "To"})
        if ev and ev.selection.rows:
            open_player(result.iloc[ev.selection.rows[0]]["player_id"])
    with st.expander("What can I ask?"):
        st.markdown(SEARCH_HELP)


SEARCH_HELP = """
- **Positions:** QB, RBs, wide receivers, tight ends
- **Stat rules:** `100+ targets`, `over 1000 rushing yards`, `passing tds > 30`, `with 10 receiving tds`,
  `target share above 25%`, `targets per game over 8`, `points per game over 15`
- **Age:** `under 25`, `older than 29`, `aged 24-26`
- **Seasons:** `last season`, `this season`, `in 2023`, `since 2022`, `2022-2024`, `last 3 seasons`
  (no season means last season; several seasons are added together unless you say `by season`)
- **Teams:** `on KC`, `Chiefs`
- **Output:** `by season` for one row per season and a line chart, `top 10` to limit the list
"""


# ============================================================== navigation
init_state()
PAGES = {
    "explore": st.Page(explore, title="Explore", icon=":material/scatter_plot:", url_path="explore", default=True),
    "player": st.Page(player_page, title="Players", icon=":material/person:", url_path="player"),
    "compare": st.Page(compare, title="Compare", icon=":material/compare_arrows:", url_path="compare"),
    "teams": st.Page(teams, title="Teams", icon=":material/shield:", url_path="teams"),
    "projections": st.Page(projections, title="Projections", icon=":material/trending_up:", url_path="projections"),
    "roster": st.Page(roster, title="Roster", icon=":material/groups:", url_path="roster"),
    "search": st.Page(search, title="Search", icon=":material/search:", url_path="search"),
}
nav = st.navigation(list(PAGES.values()), position="top")
try:
    _world = build_world(st.session_state["scoring"])
except Exception as exc:  # first run with no network, corrupt cache, ...
    st.error("The NFL data couldn't be loaded. Check your internet connection, then reload the page. "
             f"Details: {exc}")
    st.stop()
sidebar(_world)
nav.run()
