"""Custom exceptions raised for invalid user input."""


class FantasyError(Exception):
    """Base class for every error the app shows to the user."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class QueryError(FantasyError):
    """A search could not be understood, or matched nothing usable."""


class RosterError(FantasyError):
    """A player could not be added to (or removed from) a roster."""


class ComparisonError(FantasyError):
    """Players in a comparison break the size or position-group rules."""


class PlotError(FantasyError):
    """A chart or trendline could not be built from the selected data."""
