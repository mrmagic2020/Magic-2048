import copy

from src.learn.rl_strategy import RLStrategy
from src.core.board import Board


class Expectimax(RLStrategy):
    """Expectimax strategy optimized for bitboard representation."""

    def __init__(self, depth: int = 5):
        """
        Initialize the expectimax strategy.

        :param depth: The depth of the expectimax search tree.
        """
        self.depth: int = depth
        self.transposition_table = {}

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def save_path(self) -> str:
        return ""

    def save(self):
        pass

    def load(self):
        pass

    def train(self, board: Board, **kwargs):
        """Train the model."""
        print("Training the expectimax strategy... to make it more expectimax? 🤔")

    def select_action(self, board: Board) -> Board.Direction:
        """
        Select an action based on the current state.

        :param board: The current board state.
        :return: The selected action.
        """
        best_action = None
        best_score = float("-inf")
        valid_moves = board.get_valid_moves()

        for action in valid_moves:
            new_board = copy.deepcopy(board)
            moved = new_board.move(action)
            if moved:
                score = self.__expectimax(new_board.board, self.depth,
                                          is_maximizing=False)
                if score > best_score:
                    best_score = score
                    best_action = action

        board.score = Board.score_board(board.board)
        return best_action

    def __expectimax(self, board_int: int, depth: int, is_maximizing: bool) -> float:
        """
        Recursive Expectimax evaluation.

        :param board_int: The current board state as a 64-bit integer.
        :param depth: The depth of the search tree.
        :param is_maximizing: Whether the current player is maximizing.
        :return: The evaluation score of the board.
        """
        # Use board_int as the key in the transposition table
        trans_table_key = (board_int, depth, is_maximizing)
        if trans_table_key in self.transposition_table:
            return self.transposition_table[trans_table_key]

        if depth == 0 or Board.is_game_over_static(board_int):
            score = self.__evaluate_board(board_int)
            self.transposition_table[trans_table_key] = score
            return score

        if is_maximizing:
            # Maximizing player (the AI making a move)
            max_value = float('-inf')
            for move in Board.Direction:
                new_board_int = Board.execute_move(move.value, board_int)
                if new_board_int != board_int:
                    value = self.__expectimax(new_board_int, depth - 1,
                                              is_maximizing=False)
                    max_value = max(max_value, value)
            self.transposition_table[trans_table_key] = max_value
            return max_value
        else:
            # Expectation player (random tile placement)
            empty_cells = Board.get_empty_cells_from_board(board_int)
            if not empty_cells:
                score = self.__evaluate_board(board_int)
                self.transposition_table[trans_table_key] = score
                return score

            total_value = 0

            for pos in empty_cells:
                index = Board.get_index(pos[0], pos[1])
                # Skip if the position is not empty (should not happen)
                if ((board_int >> index) & 0xF) != 0:
                    continue

                # Simulate placing a 2 or 4 (represented by ranks 1 and 2)
                for tile_rank, tile_probability in [(1, 0.9), (2, 0.1)]:
                    new_board_int = board_int | (tile_rank << index)
                    value = self.__expectimax(new_board_int, depth - 1,
                                              is_maximizing=True)
                    total_value += tile_probability * value

            expected_value = total_value / len(empty_cells)
            self.transposition_table[trans_table_key] = expected_value
            return expected_value

    @staticmethod
    def __evaluate_board(board_int: int) -> float:
        """
        Evaluate the board using the heuristic function.

        :param board_int: The board state as a 64-bit integer.
        :return: The evaluation score of the board.
        """
        # Use the precomputed heuristic score tables
        score = Board.heuristic_score_table[(board_int >> 0) & Board.ROW_MASK] + \
                Board.heuristic_score_table[(board_int >> 16) & Board.ROW_MASK] + \
                Board.heuristic_score_table[(board_int >> 32) & Board.ROW_MASK] + \
                Board.heuristic_score_table[(board_int >> 48) & Board.ROW_MASK]

        # Now add corner heuristic
        # Get the corner tile ranks
        rank_tl = (board_int >> 0) & 0xF
        rank_tr = (board_int >> 12) & 0xF
        rank_bl = (board_int >> 48) & 0xF
        rank_br = (board_int >> 60) & 0xF

        corner_ranks = [rank_tl, rank_tr, rank_bl, rank_br]

        # Get the maximum tile rank on the board
        max_rank = 0
        temp_board = board_int
        for _ in range(16):
            rank = temp_board & 0xF
            if rank > max_rank:
                max_rank = rank
            temp_board >>= 4

        # If the maximum tile is in a corner, reward
        if max_rank in corner_ranks:
            # Add a bonus to the score proportional to the rank
            corner_bonus = max_rank * 18875.884099326922
            score += corner_bonus
        else:
            # Penalize if the max tile is not in a corner
            corner_penalty = -max_rank * 18875.884099326922
            score += corner_penalty

        return score
