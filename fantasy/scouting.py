"""Rules-based scouting notes for a player, written from the stats (no AI model)."""
from __future__ import annotations

import math

import pandas as pd

from .player import Player
from .projector import Projector

# Rule-of-thumb age windows where fantasy production usually holds up
PRIME_WINDOW = {"QB": (24, 34), "RB": (22, 27), "WR": (23, 29), "TE": (24, 30)}
VOLUME_STAT = {"QB": ("attempts_pg", "pass attempts"), "RB": ("opportunities_pg", "carries plus targets"),
               "WR": ("targets_pg", "targets"), "TE": ("targets_pg", "targets")}
EFFICIENCY_STAT = {"QB": ("yards_per_attempt", "yards per attempt"),
                   "RB": ("yards_per_carry", "yards per carry"),
                   "WR": ("yards_per_target", "yards per target"),
                   "TE": ("yards_per_target", "yards per target")}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def scouting_notes(player: Player, all_seasons: pd.DataFrame, projector: Projector,
                   current_season: int, scoring_name: str, min_games: int = 6) -> list[str]:
    notes: list[str] = []
    pos = player.position
    last_name = player.name.split()[-1]
    done = player.completed_seasons()
    done = done[done["games"] >= 4]
    if done.empty:
        notes.append(f"{player.name} has no completed season with four or more games, so there's "
                     "little history to judge. Lean on this season's weekly usage.")
    else:
        latest = done.iloc[-1]
        yr = int(latest["season"])
        peers = all_seasons[(all_seasons["season"] == yr) & (all_seasons["position"] == pos)
                            & (all_seasons["games"] >= min_games)]
        vcol, vname = VOLUME_STAT.get(pos, ("opportunities_pg", "opportunities"))
        if vcol in peers and len(peers):
            rank = int((peers[vcol] > latest[vcol]).sum()) + 1
            notes.append(f"In {yr}, {last_name} averaged {latest[vcol]:.1f} {vname} a game, "
                         f"{_ordinal(rank)} of {len(peers)} {pos}s with {min_games}+ games.")
        rank = int((peers["fpts_pg"] > latest["fpts_pg"]).sum()) + 1
        notes.append(f"{latest['fpts_pg']:.1f} {scoring_name} points per game ranked "
                     f"{_ordinal(rank)} at the position.")
        ecol, ename = EFFICIENCY_STAT.get(pos, (None, None))
        if ecol and ecol in peers and not math.isnan(latest.get(ecol, math.nan)):
            med = peers[ecol].median()
            word = "above" if latest[ecol] > med * 1.05 else ("below" if latest[ecol] < med * 0.95 else "close to")
            notes.append(f"Efficiency: {latest[ecol]:.1f} {ename}, {word} the {pos} median of {med:.1f}.")

    slope = projector.slope(player)
    if len(done) >= 2:
        first = int(done["season"].min())
        if slope > 0.5:
            notes.append(f"Trending up: per-game scoring has grown about {slope:.1f} points a season since {first}.")
        elif slope < -0.5:
            notes.append(f"Trending down: per-game scoring has dropped about {abs(slope):.1f} points a season since {first}.")
        else:
            notes.append(f"Steady: per-game scoring has stayed within half a point a season of its trend since {first}.")

    if "age" in player.seasons and player.seasons["age"].notna().any():
        age = float(player.seasons["age"].dropna().iloc[-1])
        lo, hi = PRIME_WINDOW.get(pos, (23, 30))
        if age > hi:
            notes.append(f"At {age:.0f}, {last_name} is past the usual {pos} production window ({lo}–{hi}), "
                         "so a decline is a real risk.")
        elif age < lo:
            notes.append(f"At {age:.0f}, {last_name} is younger than the usual {pos} production window "
                         f"({lo}–{hi}); there may be room to grow.")
        else:
            notes.append(f"At {age:.0f}, {last_name} is inside the usual {pos} production window ({lo}–{hi}).")

    cur = player.seasons[player.seasons["season"] == current_season]
    pg = projector.project_per_game(player)
    if len(cur) and cur.iloc[0]["games"] > 0:
        c = cur.iloc[0]
        line = f"{current_season} so far: {c['fpts_pg']:.1f} points per game over {int(c['games'])} games"
        if not math.isnan(pg):
            diff = c["fpts_pg"] - pg
            line += (f", {abs(diff):.1f} {'above' if diff >= 0 else 'below'} the preseason projection of {pg:.1f}.")
        notes.append(line if line.endswith(".") else line + ".")
    return notes
