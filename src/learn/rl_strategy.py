from abc import ABC, abstractmethod
from pathlib import Path

from src.core.board import Board


class RLStrategy(ABC):
    """Abstract class for reinforcement learning strategies."""

    @property
    @abstractmethod
    def is_trainable(self) -> bool:
        """Whether the strategy is trainable."""
        pass

    @property
    @abstractmethod
    def save_path(self) -> str:
        """The save path for the model."""
        pass

    @abstractmethod
    def save(self):
        """Save the model to a file."""
        pass

    @abstractmethod
    def load(self):
        """Load the model from a file."""
        pass

    @abstractmethod
    def train(self, board: Board, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def select_action(self, board: Board) -> Board.Direction:
        """Select an action based on the current state."""
        pass
