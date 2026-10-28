"""DataQuery: filtering, grouping, summary tables and the rules-based search (R4-R6, R18).

The search is deterministic - no AI. It recognises, in any order:
  positions      QB, RBs, wide receivers, tight ends ...
  stat rules     "100+ targets", "over 1000 rushing yards", "passing tds > 30",
                 "with 10 receiving tds", "target share above 25%", "... per game"
  age rules      "under 25", "older than 29", "age >= 30", "aged 24-26"
  seasons        "last season", "this season", "in 2023", "since 2022" (completed seasons),
                 "2022-2024", "last 3 seasons"   (default: last completed season)
  team           "on KC", "Chiefs"
  output         "by season" (one row per season + line chart), "top 10"
"""
from __future__ import annotations

import re

import pandas as pd

from .errors import QueryError
from .processor import PER_GAME_STATS, aggregate_players
from .scoring import ScoringFormat
from .teams import TEAM_NAMES

STAT_LABELS: dict[str, str] = {
    "fpts": "Fantasy points", "fpts_pg": "Fantasy pts / game", "games": "Games",
    "age": "Age", "targets": "Targets", "targets_pg": "Targets / game",
    "receptions": "Receptions", "receptions_pg": "Receptions / game",
    "receiving_yards": "Receiving yards", "receiving_yards_pg": "Receiving yds / game",
    "receiving_tds": "Receiving TDs", "rushing_yards": "Rushing yards",
    "rushing_yards_pg": "Rushing yds / game", "rushing_tds": "Rushing TDs",
    "carries": "Carries", "carries_pg": "Carries / game", "passing_yards": "Passing yards",
    "passing_yards_pg": "Passing yds / game", "passing_tds": "Passing TDs",
    "passing_tds_pg": "Passing TDs / game", "passing_interceptions": "Interceptions",
    "attempts": "Pass attempts", "attempts_pg": "Pass attempts / game",
    "completions": "Completions", "total_tds": "Total TDs",
    "scrimmage_yards": "Scrimmage yards", "target_share": "Target share",
    "carry_share": "Carry share", "yards_per_carry": "Yards / carry",
    "yards_per_target": "Yards / target", "catch_rate": "Catch rate",
    "yards_per_attempt": "Yards / pass attempt", "opportunities": "Opportunities",
    "opportunities_pg": "Opportunities / game", "receiving_air_yards": "Air yards",
    "receiving_yards_after_catch": "Yards after catch", "fumbles_lost": "Fumbles lost",
    "season": "Season",
}

# Columns averaged in summary tables
SUMMARY_STATS = ["age", "games", "fpts", "fpts_pg", "targets", "receptions", "receiving_yards",
                 "carries", "rushing_yards", "passing_yards", "total_tds", "opportunities_pg"]

GROUPABLE = {"position": "Position", "team": "Team", "age_bin": "Age range", "season": "Season"}

_ALIASES: dict[str, list[str]] = {
    "fpts_pg": ["fantasy points per game", "points per game", "ppg", "fpts per game"],
    "fpts": ["fantasy points", "fantasy pts", "fpts", "points"],
    "receiving_yards": ["receiving yards", "receiving yds", "rec yards", "rec yds"],
    "receiving_tds": ["receiving touchdowns", "receiving tds", "rec tds"],
    "rushing_yards": ["rushing yards", "rushing yds", "rush yards", "rush yds"],
    "rushing_tds": ["rushing touchdowns", "rushing tds", "rush tds"],
    "passing_yards": ["passing yards", "passing yds", "pass yards", "pass yds"],
    "passing_tds": ["passing touchdowns", "passing tds", "pass tds", "td passes"],
    "passing_interceptions": ["interceptions thrown", "interceptions", "ints", "picks"],
    "attempts": ["pass attempts", "passing attempts", "attempts"],
    "completions": ["completions"],
    "targets": ["targets"],
    "receptions": ["receptions", "catches"],
    "carries": ["rush attempts", "carries", "rushes"],
    "games": ["games played", "games"],
    "scrimmage_yards": ["yards from scrimmage", "scrimmage yards", "total yards"],
    "total_tds": ["total touchdowns", "total tds", "touchdowns", "tds"],
    "target_share": ["target share"],
    "carry_share": ["carry share", "rush share"],
    "yards_per_carry": ["yards per carry", "ypc"],
    "yards_per_target": ["yards per target"],
    "catch_rate": ["catch rate"],
    "opportunities": ["opportunities"],
    "receiving_air_yards": ["air yards"],
    "fumbles_lost": ["fumbles lost", "fumbles"],
}
_ALIAS_TO_COL = {a: col for col, names in _ALIASES.items() for a in names}
_STAT_RE = "|".join(sorted((re.escape(a) for a in _ALIAS_TO_COL), key=len, reverse=True))
_NUM = r"(\d+(?:\.\d+)?)\s*(%?)"

_OP_WORDS = {
    ">=": ">=", "at least": ">=", "of at least": ">=", "no less than": ">=",
    ">": ">", "over": ">", "above": ">", "more than": ">", "greater than": ">",
    "<=": "<=", "at most": "<=", "no more than": "<=",
    "<": "<", "under": "<", "below": "<", "less than": "<", "fewer than": "<",
    "=": "=", "exactly": "=",
}
_OP_RE = "|".join(sorted((re.escape(o) for o in _OP_WORDS), key=len, reverse=True))

_POSITION_WORDS = {
    "qb": "QB", "qbs": "QB", "quarterback": "QB", "quarterbacks": "QB",
    "rb": "RB", "rbs": "RB", "running back": "RB", "running backs": "RB",
    "wr": "WR", "wrs": "WR", "wide receiver": "WR", "wide receivers": "WR",
    "receiver": "WR", "receivers": "WR",
    "te": "TE", "tes": "TE", "tight end": "TE", "tight ends": "TE",
}
_NICKNAMES = {name.split()[-1].lower(): abbr for abbr, name in TEAM_NAMES.items()}


def label(col: str) -> str:
    return STAT_LABELS.get(col, col.replace("_", " ").capitalize())


def _compare(series: pd.Series, op: str, value: float) -> pd.Series:
    return {">=": series >= value, ">": series > value, "<=": series <= value,
            "<": series < value, "=": series == value}[op]


class DataQuery:
    def __init__(self, filters: dict | None = None, group_by: list[str] | None = None,
                 stat: str = "fpts"):
        self.filters: dict = filters or {}
        self.group_by: list[str] = group_by or []
        self.stat = stat
        self.conditions: list[tuple[str, str, float]] = []
        self.by_season = False
        self.limit: int | None = None
        self.text = ""

    # -------------------------------------------------------- filter + group
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        f = self.filters
        out = df
        if f.get("seasons"):
            lo, hi = f["seasons"]
            out = out[out["season"].between(lo, hi)]
        if f.get("positions"):
            out = out[out["position"].isin(f["positions"])]
        if f.get("teams"):
            out = out[out["team"].isin(f["teams"])]
        if f.get("min_games"):
            out = out[out["games"] >= f["min_games"]]
        if f.get("age"):
            lo, hi = f["age"]
            out = out[out["age"].between(lo, hi)]
        if f.get("player_ids") is not None:
            out = out[out["player_id"].isin(f["player_ids"])]
        for col, op, val in self.conditions:
            if col not in out:
                raise QueryError(f"The stat '{label(col)}' isn't available here.")
            out = out[_compare(out[col], op, val)]
        return out

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        stats = [c for c in SUMMARY_STATS if c in df]
        if df.empty:
            return pd.DataFrame(columns=self.group_by + ["players"] + stats)
        if not self.group_by:
            row = df[stats].mean().round(2).to_frame().T
            row.insert(0, "players", df["player_id"].nunique())
            return row
        g = df.groupby(self.group_by, observed=True)
        out = g[stats].mean().round(2)
        out.insert(0, "players", g["player_id"].nunique())
        return out.reset_index()

    # ---------------------------------------------------------------- search
    def parse(self, text: str, last_completed: int = 2025, current: int = 2026) -> None:
        """Turn a plain-English question into filters + conditions. Raises QueryError."""
        if not text or not text.strip():
            raise QueryError("Type a question first, for example: WRs under 25 with 100+ targets last season.")
        self.text = text.strip()
        s = " " + text.lower().replace(",", " ") + " "
        self.filters, self.conditions, self.by_season, self.limit = {}, [], False, None

        # stat conditions (removed from the text once matched)
        patterns = [
            (rf"{_NUM}\s*\+\s*({_STAT_RE})(\s+per game)?", "num_plus"),
            (rf"(?:with\s+)?({_OP_RE})\s+{_NUM}\s+({_STAT_RE})(\s+per game)?", "op_num_stat"),
            (rf"({_STAT_RE})(\s+per game)?\s*(?:of\s+)?({_OP_RE})\s*{_NUM}", "stat_op_num"),
            (rf"with\s+{_NUM}\s+({_STAT_RE})(\s+per game)?", "with_num_stat"),
        ]
        for pat, kind in patterns:
            for m in list(re.finditer(pat, s)):
                g = m.groups()
                if kind == "num_plus":
                    num, pct, alias, pg, op = g[0], g[1], g[2], g[3], ">="
                elif kind == "op_num_stat":
                    op, num, pct, alias, pg = _OP_WORDS[g[0]], g[1], g[2], g[3], g[4]
                elif kind == "stat_op_num":
                    alias, pg, op, num, pct = g[0], g[1], _OP_WORDS[g[2]], g[3], g[4]
                else:
                    num, pct, alias, pg, op = g[0], g[1], g[2], g[3], ">="
                col = self._resolve(alias, bool(pg))
                value = float(num) / 100 if (pct or (col.endswith("_share") or col == "catch_rate")
                                             and float(num) > 1) else float(num)
                self.conditions.append((col, op, value))
                s = s[:m.start()] + " " * (m.end() - m.start()) + s[m.end():]

        leftover = re.search(r"(\d+)\s*\+\s*([a-z]+(?:\s[a-z]+)?)", s)
        if leftover:
            raise QueryError(f"I don't know the stat '{leftover.group(2)}'. Try targets, receptions, "
                             "rushing yards, passing tds, fantasy points or points per game.")

        # age
        m = re.search(r"aged?\s+(\d{2})\s*(?:-|to)\s*(\d{2})", s)
        if m:
            self.filters["age"] = (float(m.group(1)), float(m.group(2)) + 0.99)
        else:
            m = re.search(r"(?:age\s*)?(under|younger than|below|less than|<|over|older than|above|>|at least|>=)\s*(\d{2})\b(?!\s*\+)", s)
            if m:
                op, n = m.group(1), float(m.group(2))
                young = op in ("under", "younger than", "below", "less than", "<")
                self.filters["age"] = (0.0, n - 0.01) if young else ((n, 60.0) if op in ("at least", ">=") else (n + 0.01, 60.0))

        # seasons
        self.filters["seasons"] = self._parse_seasons(s, last_completed, current)

        # positions
        found = []
        for word, pos in sorted(_POSITION_WORDS.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(word)}\b", s) and pos not in found:
                found.append(pos)
                s = re.sub(rf"\b{re.escape(word)}\b", " ", s)
        if found:
            self.filters["positions"] = found

        # team
        teams = [abbr for nick, abbr in _NICKNAMES.items() if re.search(rf"\b{nick}\b", s)]
        m = re.search(r"\b(?:on|for)\s+(?:the\s+)?([a-z]{2,3})\b", s)
        if m and m.group(1).upper() in TEAM_NAMES:
            teams.append(m.group(1).upper())
        if teams:
            self.filters["teams"] = sorted(set(teams))

        if re.search(r"by season|each season|per season|season by season|over time|by year", s):
            self.by_season = True
        m = re.search(r"\b(?:top|best)\s+(\d+)", s)
        if m:
            self.limit = int(m.group(1))

        self.stat = self.conditions[0][0] if self.conditions else "fpts"
        if not found and not self.conditions and "teams" not in self.filters:
            raise QueryError("I couldn't find a position or a stat rule in that question. "
                             "Try something like: RBs with 1000+ rushing yards since 2023.")

    @staticmethod
    def _resolve(alias: str, per_game: bool) -> str:
        col = _ALIAS_TO_COL[alias.strip()]
        if per_game and (col in PER_GAME_STATS or col in ("fpts", "opportunities")):
            return f"{col}_pg"
        return col

    @staticmethod
    def _parse_seasons(s: str, last: int, current: int) -> tuple[int, int]:
        if m := re.search(r"(20\d\d)\s*(?:-|to|through|thru)\s*(20\d\d)", s):
            lo, hi = sorted((int(m.group(1)), int(m.group(2))))
            return (lo, hi)
        if m := re.search(r"since\s+(20\d\d)", s):
            return (int(m.group(1)), last)
        if m := re.search(r"last\s+(\d)\s+(?:seasons|years)", s):
            return (last - int(m.group(1)) + 1, last)
        if m := re.search(r"\b(?:in\s+)?(20\d\d)\b", s):
            y = int(m.group(1))
            return (y, y)
        if re.search(r"this (?:season|year)|so far|current season", s):
            return (current, current)
        return (last, last)

    def run(self, df: pd.DataFrame, scoring: ScoringFormat) -> pd.DataFrame:
        """Execute a parsed search against the processed player-season table."""
        base = DataQuery(filters={k: v for k, v in self.filters.items() if k != "age"})
        rows = base.apply(df)
        lo, hi = self.filters["seasons"]
        if lo != hi and not self.by_season:
            rows = aggregate_players(rows, scoring)
        age_q = DataQuery(filters={"age": self.filters.get("age")})
        age_q.conditions = self.conditions
        out = age_q.apply(rows)
        if self.by_season:
            keep = out["player_id"].unique()
            out = rows[rows["player_id"].isin(keep)] if lo != hi else out
            out = out.sort_values(["season", self.stat], ascending=[True, False])
        else:
            out = out.sort_values(self.stat, ascending=False)
            if self.limit:
                out = out.head(self.limit)
        if out.empty:
            raise QueryError("No players match every rule. Loosen one condition (for example a lower number) and search again.")
        return out

    def describe(self) -> list[str]:
        """Human-readable chips for what the search understood."""
        chips = []
        if self.filters.get("positions"):
            chips.append(" / ".join(self.filters["positions"]))
        if self.filters.get("teams"):
            chips.append(", ".join(self.filters["teams"]))
        if self.filters.get("age"):
            lo, hi = self.filters["age"]
            chips.append(f"Age under {hi + 0.01:.0f}" if lo == 0 else
                         (f"Age over {lo - 0.01:.0f}" if hi == 60 else f"Age {lo:.0f}–{hi:.0f}"))
        for col, op, val in self.conditions:
            shown = f"{val:.0%}" if col.endswith("_share") or col == "catch_rate" else f"{val:g}"
            chips.append(f"{label(col)} {op.replace('>=', '≥').replace('<=', '≤')} {shown}")
        lo, hi = self.filters.get("seasons", (None, None))
        if lo:
            chips.append(f"{lo}" if lo == hi else f"{lo}–{hi}" + (" by season" if self.by_season else " combined"))
        if self.limit:
            chips.append(f"Top {self.limit}")
        return chips

    def chart_spec(self, result: pd.DataFrame) -> dict:
        """Pick the chart that best fits the result."""
        stat_cols = list(dict.fromkeys(c for c, _, _ in self.conditions))
        if self.by_season and result["season"].nunique() > 1:
            return {"chart_type": "line", "x": "season", "y": self.stat, "group_by": "name"}
        if len(stat_cols) >= 2:
            return {"chart_type": "scatter", "x": stat_cols[0], "y": stat_cols[1], "group_by": "position"}
        y = self.stat if self.stat != "fpts" or not stat_cols else stat_cols[0]
        return {"chart_type": "bar", "x": "name", "y": y, "group_by": "position"}
