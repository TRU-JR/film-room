"""Roster: the user's fantasy team, lineup slots, projections, JSON save/load (R16, R17)."""
from __future__ import annotations

import json
import math
import os
import re
from typing import Callable

from .errors import RosterError
from .player import Player
from .projector import Projector
from .scoring import ScoringFormat

DEFAULT_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "D/ST": 1, "BENCH": 6}
FLEX_ELIGIBLE = {"RB", "WR", "TE"}
STARTING_SLOTS = ["QB", "RB", "WR", "TE", "FLEX", "D/ST"]


def _home_slot(position: str) -> str:
    return "D/ST" if position == "DST" else position


class Roster:
    def __init__(self, name: str = "My team", slots: dict[str, int] | None = None,
                 scoring: ScoringFormat | None = None):
        self.name = name
        self.slots = dict(slots or DEFAULT_SLOTS)
        self.players: list[Player] = []
        self.assignments: dict[str, str] = {}  # player_id -> slot
        self.scoring = scoring or ScoringFormat("PPR")

    # -------------------------------------------------------------- lineup
    def open_slots(self, slot: str) -> int:
        used = sum(1 for s in self.assignments.values() if s == slot)
        return self.slots.get(slot, 0) - used

    def slot_for(self, p: Player) -> str:
        home = _home_slot(p.position)
        if home not in self.slots:
            raise RosterError(f"{p.position} isn't a position this lineup uses.")
        if self.open_slots(home) > 0:
            return home
        if p.position in FLEX_ELIGIBLE and self.open_slots("FLEX") > 0:
            return "FLEX"
        if self.open_slots("BENCH") > 0:
            return "BENCH"
        raise RosterError(f"No open slot for {p.name}: the {home} spots, FLEX and bench are full. "
                          "Remove a player first.")

    def add(self, p: Player, slot: str | None = None) -> str:
        if p in self.players:
            raise RosterError(f"{p.name} is already on {self.name}.")
        if slot is None:
            slot = self.slot_for(p)
        else:
            self._check_slot(p, slot)
        self.players.append(p)
        self.assignments[p.player_id] = slot
        return slot

    def _check_slot(self, p: Player, slot: str) -> None:
        if slot not in self.slots:
            raise RosterError(f"Unknown slot '{slot}'.")
        ok = slot in (_home_slot(p.position), "BENCH") or (slot == "FLEX" and p.position in FLEX_ELIGIBLE)
        if not ok:
            raise RosterError(f"A {p.position} can't play the {slot} slot.")
        if self.open_slots(slot) <= 0:
            raise RosterError(f"The {slot} slot is full.")

    def move(self, p: Player, slot: str) -> None:
        if p not in self.players:
            raise RosterError(f"{p.name} isn't on {self.name}.")
        old = self.assignments.pop(p.player_id)
        try:
            self._check_slot(p, slot)
        except RosterError:
            self.assignments[p.player_id] = old
            raise
        self.assignments[p.player_id] = slot

    def remove(self, p: Player) -> None:
        if p not in self.players:
            raise RosterError(f"{p.name} isn't on {self.name}.")
        self.players.remove(p)
        self.assignments.pop(p.player_id, None)

    def starters(self) -> list[Player]:
        return [p for p in self.players if self.assignments[p.player_id] != "BENCH"]

    def lineup(self) -> list[tuple[str, Player | None]]:
        """Every slot in display order, with None for empty ones."""
        rows: list[tuple[str, Player | None]] = []
        for slot in STARTING_SLOTS + ["BENCH"]:
            filled = [p for p in self.players if self.assignments[p.player_id] == slot]
            for i in range(self.slots.get(slot, 0)):
                rows.append((slot, filled[i] if i < len(filled) else None))
        return rows

    def total_projection(self, proj: Projector | Callable[[Player], float]) -> float:
        """Sum of starters' projections (bench excluded)."""
        fn = proj.project if isinstance(proj, Projector) else proj
        vals = [fn(p) for p in self.starters()]
        return round(sum(v for v in vals if not math.isnan(v)), 1)

    # ------------------------------------------------------------- storage
    def to_dict(self) -> dict:
        return {"name": self.name, "scoring": self.scoring.name, "slots": self.slots,
                "players": [{"player_id": p.player_id, "name": p.name, "position": p.position,
                             "team": p.team, "slot": self.assignments[p.player_id]}
                            for p in self.players]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str, lookup: Callable[[str], Player] | None = None) -> "Roster":
        """Rebuild a roster from JSON text. `lookup` turns a player_id into a full Player."""
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or not isinstance(data.get("players", []), list):
                raise ValueError("not a roster file")
        except (json.JSONDecodeError, ValueError) as exc:
            raise RosterError(f"That file isn't a saved roster ({exc}).") from exc
        roster = cls(data.get("name", "My team"), data.get("slots"),
                     ScoringFormat(data.get("scoring", "PPR")))
        for item in data["players"]:
            try:
                p = lookup(item["player_id"]) if lookup else None
            except KeyError:
                p = None
            if p is None:
                p = Player(item["player_id"], item["name"], item["position"], item["team"])
            roster.players.append(p)
            roster.assignments[p.player_id] = item["slot"]
        return roster

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path: str, lookup: Callable[[str], Player] | None = None) -> "Roster":
        """Read a roster saved with save()."""
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            raise RosterError(f"Couldn't read the roster file {os.path.basename(path)}: {exc}") from exc
        return cls.from_json(text, lookup)

    @staticmethod
    def file_name(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "roster"
        return f"{slug}.json"
