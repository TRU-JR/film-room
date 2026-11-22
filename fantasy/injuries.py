"""Injury status: who is hurt right now, and which past seasons an injury cost a player.

Built from the injury table DataLoader caches (one row per player per week hurt):
  source "report"   the official weekly injury report (Out / Doubtful / Questionable)
  source "reserve"  a reserve list on the weekly roster (IR, PUP, NFI)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# most serious first; used to pick one status when a player has several in a week
SEVERITY = ["IR", "PUP", "NFI", "Out", "Doubtful", "Questionable"]
STATUS_TEXT = {"IR": "Injured reserve", "PUP": "PUP list", "NFI": "Non-football injury list",
               "Out": "Out", "Doubtful": "Doubtful", "Questionable": "Questionable"}
RED, ORANGE = "🔴", "🟠"


@dataclass(frozen=True)
class Injury:
    status: str          # one of SEVERITY
    injury: str | None   # body part from the injury report, e.g. "Knee"
    week: int

    @property
    def dot(self) -> str:
        return ORANGE if self.status == "Questionable" else RED

    @property
    def text(self) -> str:
        part = f" ({self.injury.lower()})" if self.injury else ""
        return f"{STATUS_TEXT[self.status]}{part}"

    @property
    def likely_out(self) -> bool:
        return self.status != "Questionable"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["player_id", "season", "week", "team", "source", "status", "injury"])
    out = df.copy()
    rest = out["injury"].astype(str).str.contains("not injury related", case=False, na=False)
    out = out[~rest]   # rest days and personal matters aren't injuries
    out["injury"] = out["injury"].where(out["injury"].notna() & (out["injury"].astype(str).str.strip() != ""))
    out["rank"] = out["status"].map({s: i for i, s in enumerate(SEVERITY)}).fillna(len(SEVERITY))
    return out


def current_injuries(injuries: pd.DataFrame, season: int) -> dict[str, Injury]:
    """Players hurt as of the latest data: on their team's most recent injury report,
    or on a reserve list in the latest weekly roster."""
    df = _clean(injuries)
    df = df[df["season"] == season]
    if df.empty:
        return {}
    rep = df[df["source"] == "report"]
    if len(rep):
        latest = rep.groupby("team")["week"].transform("max")   # teams on a bye keep last week's report
        rep = rep[rep["week"] == latest]
    res = df[df["source"] == "reserve"]
    if len(res):
        res = res[res["week"] == res["week"].max()]
    now = pd.concat([rep, res]).sort_values(["rank", "week"], ascending=[True, False])
    # reserve-list rows carry no body part: borrow the player's latest reported one
    last_part = (df.dropna(subset=["injury"]).sort_values("week").groupby("player_id")["injury"].last())
    out: dict[str, Injury] = {}
    for pid, row in now.drop_duplicates("player_id").set_index("player_id").iterrows():
        part = row["injury"] if isinstance(row["injury"], str) else last_part.get(pid)
        out[pid] = Injury(row["status"], part, int(row["week"]))
    return out


def season_injuries(injuries: pd.DataFrame, player_id: str) -> pd.DataFrame:
    """One row per season in which an injury kept the player out:
    season, weeks_out (Out on the report or on IR/PUP/NFI), injury, reserve (bool), note."""
    df = _clean(injuries)
    df = df[df["player_id"] == player_id]
    missed = df[(df["source"] == "reserve") | (df["status"] == "Out")]
    rows = []
    for season, part in missed.groupby("season"):
        weeks = int(part["week"].nunique())
        year = df[df["season"] == season].sort_values("week")
        serious = year[year["status"].isin(["Out", "Doubtful"])]["injury"].dropna()
        parts = list(dict.fromkeys(serious if len(serious) else year["injury"].dropna()))[:2]
        injury = ", ".join(parts) if parts else None
        lists = sorted(set(part.loc[part["source"] == "reserve", "status"]), key=SEVERITY.index)
        where = f", on {' and '.join(lists)}" if lists else ""
        what = f" ({injury.lower()})" if injury else ""
        rows.append({"season": int(season), "weeks_out": weeks, "injury": injury, "reserve": bool(lists),
                     "note": f"Injured{what}: out {weeks} week{'s' if weeks != 1 else ''}{where}"})
    return pd.DataFrame(rows, columns=["season", "weeks_out", "injury", "reserve", "note"])
