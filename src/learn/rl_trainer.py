import enum

from learn.rl_strategy import RLStrategy
from learn.random_strategy import RandomStrategy
from learn.expectimax import Expectimax
from core.board import Board


class RLTrainer:
    """
    Context class for training a reinforcement learning strategy.

    :var strategy: The reinforcement learning strategy.
    """

    class Strategy(enum.Enum):
        """Enumeration of strategies."""
        RANDOM = 0
        EXPECTIMAX = 1

    def __init__(self, strategy: RLStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: Strategy):
        """Set the strategy."""
        if strategy == self.Strategy.RANDOM:
            self.strategy = RandomStrategy()
        elif strategy == self.Strategy.EXPECTIMAX:
            self.strategy = Expectimax()
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def select_action(self, board: Board) -> Board.Direction:
        """
        Select an action based on the current state.

        :param board: The current board state.
        :return: The selected action.
        """
        return self.strategy.select_action(board)

    def train(self, board: Board, **kwargs):
        """
        Train the model.

        :param board: The current board state.
        :param kwargs:
        """
        self.strategy.train(board, **kwargs)

    @property
    def is_trainable(self) -> bool:
        """Whether the strategy is trainable."""
        return self.strategy.is_trainable

    def save(self):
        """Save the model to a file."""
        self.strategy.save()

    def load(self):
        """Load the model from a file."""
        self.strategy.load()
