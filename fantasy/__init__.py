"""NFL fantasy analytics: data, scoring, projections, search, plots and rosters."""
from .comparison import Comparison
from .data_loader import DataLoader
from .errors import ComparisonError, FantasyError, PlotError, QueryError, RosterError
from .player import Player
from .plots import PlotBuilder
from .processor import StatProcessor, aggregate_players
from .projector import Projector
from .query import DataQuery
from .roster import Roster
from .scoring import ScoringFormat

__all__ = [
    "Comparison", "DataLoader", "ComparisonError", "FantasyError", "PlotError", "QueryError",
    "RosterError", "Player", "PlotBuilder", "StatProcessor", "aggregate_players", "Projector",
    "DataQuery", "Roster", "ScoringFormat",
]
