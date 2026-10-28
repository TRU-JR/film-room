import math

import pytest

from fantasy import Player, Projector, Roster, RosterError, ScoringFormat


def make(pid, pos):
    return Player(pid, f"Player {pid}", pos, "KC")


def test_players_fill_home_then_flex_then_bench():
    r = Roster("Test")
    assert r.add(make("r1", "RB")) == "RB"
    assert r.add(make("r2", "RB")) == "RB"
    assert r.add(make("r3", "RB")) == "FLEX"
    assert r.add(make("r4", "RB")) == "BENCH"
    assert r.add(make("q1", "QB")) == "QB"
    assert r.add(make("q2", "QB")) == "BENCH"     # QBs can't play FLEX
    assert r.add(make("DST-KC", "DST")) == "D/ST"


def test_full_roster_raises():
    r = Roster("Tiny", slots={"QB": 1, "BENCH": 0})
    r.add(make("q1", "QB"))
    with pytest.raises(RosterError, match="No open slot"):
        r.add(make("q2", "QB"))


def test_duplicate_player_raises():
    r = Roster()
    p = make("w1", "WR")
    r.add(p)
    with pytest.raises(RosterError, match="already"):
        r.add(p)


def test_move_rules():
    r = Roster()
    qb, wr = make("q1", "QB"), make("w1", "WR")
    r.add(qb)
    r.add(wr)
    r.move(wr, "FLEX")
    assert r.assignments["w1"] == "FLEX"
    with pytest.raises(RosterError):
        r.move(qb, "FLEX")
    assert r.assignments["q1"] == "QB"   # failed move leaves the player where he was


def test_remove():
    r = Roster()
    p = make("t1", "TE")
    r.add(p)
    r.remove(p)
    assert r.players == [] and r.assignments == {}
    with pytest.raises(RosterError):
        r.remove(p)


def test_total_projection_skips_bench(players):
    r = Roster()
    r.add(players["WR1"])
    r.add(players["RB1"])
    bench = players["QB1"]
    r.add(make("q0", "QB"))         # takes the QB slot
    r.add(bench)                    # goes to the bench
    proj = Projector(target_season=2025)
    expected = proj.project(players["WR1"]) + proj.project(players["RB1"])
    assert r.total_projection(proj) == pytest.approx(expected, abs=0.2)   # q0 has no games -> NaN skipped


def test_save_and_load_round_trip(tmp_path, players):
    r = Roster("Sunday Squad", scoring=ScoringFormat("Half-PPR"))
    r.add(players["WR1"])
    r.add(players["RB1"], slot="FLEX")
    path = tmp_path / Roster.file_name(r.name)
    r.save(str(path))
    assert path.name == "sunday-squad.json"

    loaded = Roster.load(str(path), lookup=lambda pid: players[pid])
    assert loaded.name == "Sunday Squad"
    assert loaded.scoring.name == "Half-PPR"
    assert [p.player_id for p in loaded.players] == ["WR1", "RB1"]
    assert loaded.assignments == {"WR1": "WR", "RB1": "FLEX"}
    assert loaded.players[0].seasons.equals(players["WR1"].seasons)


def test_load_without_lookup_and_bad_file(tmp_path):
    r = Roster("Stub")
    r.add(make("w9", "WR"))
    path = tmp_path / "stub.json"
    r.save(str(path))
    loaded = Roster.load(str(path))
    assert loaded.players[0].name == "Player w9"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(RosterError):
        Roster.load(str(bad))


def test_json_text_round_trip(players):
    r = Roster("Browser Team")
    r.add(players["QB1"])
    r.add(players["TE1"])
    loaded = Roster.from_json(r.to_json(), lookup=lambda pid: players[pid])
    assert loaded.to_dict() == r.to_dict()
    with pytest.raises(RosterError):
        Roster.from_json('["not", "a", "roster"]')
