import enum

from src.learn.rl_strategy import RLStrategy
from src.learn.random_strategy import RandomStrategy
from src.learn.expectimax import Expectimax
from src.learn.dqn_strategy import DQNStrategy
from src.core.board import Board


class RLTrainer:
    """
    Context class for training a reinforcement learning strategy.

    :var strategy: The reinforcement learning strategy.
    """

    class Strategy(enum.Enum):
        """Enumeration of strategies."""
        RANDOM = 0
        EXPECTIMAX = 1
        DQN = 2

    def __init__(self, strategy: RLStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: Strategy):
        """Set the strategy."""
        if strategy == self.Strategy.RANDOM:
            self.strategy = RandomStrategy()
        elif strategy == self.Strategy.EXPECTIMAX:
            self.strategy = Expectimax()
        elif strategy == self.Strategy.DQN:
            self.strategy = DQNStrategy(state_size=16, action_size=4)
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
