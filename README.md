# Film Room: NFL fantasy analytics

A Python + Streamlit web app that helps you find the players and teams most likely to score fantasy points.
It follows the Features and Classes sheets in the development plan: every Essential and Complete feature is implemented.

## Put it on the web (free, nothing to install)

The app runs on Streamlit Community Cloud, so anyone with the link can use it in a browser.

**Part 1: put the code on GitHub**

1. Unzip `film_room.zip`. Open the `fantasy_app` folder; you should see `app.py`, `requirements.txt`,
   `fantasy`, `data_cache`, `static`, `tests` and `.streamlit`.
2. Go to github.com, sign in, and select **+** (top right) > **New repository**.
3. Name it (for example `film-room`), choose **Public**, leave every other box unchecked, and select **Create repository**.
4. On the empty repository page, select the **uploading an existing file** link.
5. Select everything inside the `fantasy_app` folder (Ctrl + A) and drag it onto the GitHub page.
   Drag the *contents*, not the `fantasy_app` folder itself, so `app.py` ends up at the top level.
6. Wait for the file list to finish, then select **Commit changes**.
7. Check the repository's file list shows a `.streamlit` folder. If it's missing, select **Add file** >
   **Create new file**, name it `.streamlit/config.toml`, paste the contents of that file from your computer, and
   commit.

**Part 2: deploy it**

1. Go to share.streamlit.io and select **Continue with GitHub**. Approve the access request.
2. Select **Create app** (top right), then **Yup, I have an app**.
3. Fill in: Repository = `your-username/film-room`, Branch = `main`, Main file path = `app.py`.
4. Optional: type a short name in **App URL**, such as `film-room-nfl`.
5. Select **Deploy**. The first build takes a few minutes while it installs the packages.
6. Copy the link from your browser's address bar and share it with your team.

**After it's live**

- To change the code, edit or re-upload files on GitHub. The app updates on its own within a minute or two.
- If nobody opens the app for a while, it goes to sleep. The next visitor selects the wake-up button and waits
  about a minute, so open it a few minutes before a demo.
- Rosters live in your browser tab. Select **Download roster file** to keep one, and use **Open roster files**
  to load it again later.
- **Download latest data** in the sidebar refreshes the stats for everyone using the app.

## Run it on your own computer (optional)

You need Python 3.10 or newer. Everything installs into a `.venv` folder inside the project.

| | Windows | macOS / Linux |
|---|---|---|
| First-time setup | double-click `setup.bat` | `./setup.sh` |
| Start the app | double-click `run.bat` | `./run.sh` |
| Run the tests | `.venv\Scripts\python -m pytest` | `.venv/bin/python -m pytest` |

A cache of nflverse data (2021–2025 plus 2026 through week 7) ships in `data_cache/`, so the app starts quickly
and works offline.

## Pages

| Page | What it does | Req. |
|---|---|---|
| Explore | Sidebar filters (season range, position, minimum games, age range, active only); chart builder with scatter, histogram, box-and-whisker and line charts, user-chosen axes, grouping, and linear / polynomial / exponential trendlines with R²; summary table of averages by position, team, age range or season | R4–R6, R10, R11 |
| Players | Career and per-season tables, per-game table, current-season game log, trend chart with projection and likely range, rules-based scouting notes | R7, R12–R14 |
| Compare | Up to six players from one position on any stat, season by season | R8 |
| Teams | Offense and defense unit rankings, head-to-head comparison, fantasy points allowed by position, D/ST fantasy points | R9 |
| Projections | 2026 projections (regression, rookie x 1.10 rule), growth flag, blend with the 2026 season so far, and a 2025 backtest showing accuracy | R12–R14, implicit 4 |
| Roster | QB / RB / WR / TE / FLEX / D/ST / bench slots, season or single-game projections with opponent matchup factors, multiple rosters, download and open roster files (JSON) | R16, R17 |
| Search | Plain-English, rules-based search, e.g. "WRs under 25 with 100+ targets last season", with an automatically chosen chart | R18 |

The scoring format (PPR, half-PPR, standard) in the sidebar updates every page (R15).

## Code layout

```
app.py                 Streamlit pages and session state (roster, scoring format, filters)
ui.py                  Theme CSS and small HTML helpers
fantasy/
  data_loader.py       DataLoader      - download nflverse data, cache to CSV, active rosters
  scoring.py           ScoringFormat   - stat weights -> fantasy points, D/ST scoring
  processor.py         StatProcessor   - age, age bins, per-game stats, fantasy points
  player.py            Player          - one player's career
  projector.py         Projector       - regression projections, rookie rule, growth flag, backtest
  query.py             DataQuery       - filters, summaries, rules-based search parser
  comparison.py        Comparison      - players (max 6, same position) and team units
  plots.py             PlotBuilder     - Plotly charts and trendline fitting
  roster.py            Roster          - lineup slots, projections, JSON save/load
  teams.py             team unit tables, D/ST points, matchup factors
  scouting.py          rules-based scouting notes
  errors.py            QueryError, RosterError, ComparisonError, PlotError
tests/                 pytest suite
```

## Method notes

- **Fantasy points** are computed from raw stats (4 pts per passing TD, 0.04 per passing yard, -2 per interception,
  0.1 per rushing/receiving yard, 6 per rushing/receiving TD, -2 per fumble lost, 2 per two-point conversion,
  reception value 1 / 0.5 / 0). PPR totals match nflverse's own numbers exactly.
- **Projections**: weighted linear regression of points per game on season (weights = games played), one season
  ahead, x 17 games, held between 0 and 125% of the player's best season. One-season players: points per game
  x 1.10 x 17. The full-season outlook adds points scored so far to remaining games at a rate that moves toward
  this season's pace as games are played (weight g / (g + 4)).
- **Matchup factor** (single-game view): points a defense allowed to the position per game (last season plus this
  season, this season counted twice) divided by the league average, kept between 0.8 and 1.2.
- **Scouting notes** are written by fixed rules from the stats; no AI model runs inside the app.

Data: [nflverse](https://github.com/nflverse/nflverse-data). Fonts: Saira Condensed and IBM Plex Sans (SIL Open Font
License, bundled in `static/fonts`).
