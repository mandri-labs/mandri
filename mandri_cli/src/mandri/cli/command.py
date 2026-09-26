"""CLI command base class owning a Rich console."""

from abc import ABC, abstractmethod

from rich.console import Console


class Command(ABC):
    """CLI command with its own Rich console."""

    def __init__(self) -> None:
        self._console = Console(log_path=False, highlighter=None)

    @abstractmethod
    def run(self) -> int:
        """Execute the command and return its exit code."""
