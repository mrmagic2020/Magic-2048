import random

from learn.rl_strategy import RLStrategy
from core.board import Board


class RandomStrategy(RLStrategy):
    """Random strategy for reinforcement learning."""

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def save_path(self) -> str:
        return ""

    def save(self): pass

    def load(self): pass

    def train(self, board: Board, **kwargs):
        """Train the model."""
        print("Training the random strategy... to make it more random? 🤔")

    def select_action(self, board: Board) -> Board.Direction:
        """Select a random action."""
        return random.choice(list(Board.Direction))
