import json
import enum
import random
from pathlib import Path


class Board:
    """A class to represent the board of the 2048 game using bitboard representation."""
    state_path = Path.home() / ".2048-AI-tmp"
    highscore_path = Path.home() / ".2048-AI-highscore"

    class Direction(enum.Enum):
        """An enumeration of the possible directions."""
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3

    def __init__(self, size: int = 4):
        """Initialize the board with the given size."""
        self.size: int = size
        self.board_int: int = 0  # 64-bit integer representing the board
        self.score: int = 0
        self.won: bool = False
        self.over: bool = False

        # Initialize random tiles on the board
        self.place_random_tile()
        self.place_random_tile()

    def load_highscore(self) -> int:
        """
        Load the highscore from a file.

        :return: The highscore.
        """
        if not self.highscore_path.exists():
            return 0

        try:
            with open(self.highscore_path, "r", encoding="utf-8") as file:
                return int(file.read())
        except Exception as e:
            # print(f"Error loading highscore: {e}")
            return 0

    def save_highscore(self) -> bool:
        """
        Save the highscore to a file.

        :return: True if the highscore was saved successfully, False otherwise.
        """
        curr_highscore = self.load_highscore()
        if self.score > curr_highscore:
            try:
                with open(self.highscore_path, "w", encoding="utf-8") as file:
                    # noinspection PyTypeChecker
                    json.dump(self.score, file)
            except Exception as e:
                print(f"Error saving highscore: {e}")
                return False
        return True

    def save_state(self) -> bool:
        """
        Save the current state of the board to a temporary file.

        :return: True if the state was saved successfully, False otherwise.
        """
        game_state = {
            "size": self.size,
            "board": self.board_int,
            "score": self.score,
            "won": self.won,
            "over": self.over
        }

        try:
            with open(self.state_path, "w", encoding="utf-8") as file:
                # noinspection PyTypeChecker
                json.dump(game_state, file)
        except Exception as e:
            print(f"Error saving game state: {e}")
            return False
        return True

    def load_state(self) -> bool:
        """
        Load the saved state of the board from a temporary file.

        :return: True if the state was loaded successfully, False otherwise.
        """
        if not self.state_path.exists():
            return False

        try:
            with open(self.state_path, "r", encoding="utf-8") as file:
                game_state = json.load(file)
                self.size = game_state["size"]
                self.board_int = game_state["board"]
                self.score = game_state["score"]
                self.won = game_state["won"]
                self.over = game_state["over"]
            return True
        except Exception as e:
            print(f"Error loading game state: {e}")
            return False

    @staticmethod
    def clear_state():
        """Clear the saved state of the board."""
        try:
            Board.state_path.unlink()
        except Exception as e:
            print(f"Error clearing game state: {e}")

    @staticmethod
    def get_index(i: int, j: int) -> int:
        """
        Get the bit index for position (i, j).

        :param i: The row index.
        :param j: The column index.
        :return: The bit index.
        """
        return 16 * i + 4 * j

    def get_tile(self, i: int, j: int, true_value=False) -> int:
        """
        Get the tile at position (i, j).

        :param i: The row index.
        :param j: The column index.
        :param true_value: Whether to return the true value of the tile or the exponent.
        :return: The value of the tile.
        """
        index = self.get_index(i, j)
        value = (self.board_int >> index) & 0xF
        return (2 ** value if value != 0 else 0) if true_value else value

    def set_tile(self, i: int, j: int, value: int):
        """
        Set the tile at position (i, j).

        :param i: The row index.
        :param j: The column index.
        :param value: The value of the tile.
        """
        index = self.get_index(i, j)
        self.board_int &= ~(0xF << index)  # Clear the tile
        self.board_int |= (value & 0xF) << index  # Set the new value

    def place_random_tile(self):
        """Place a random tile on the board."""
        empty_cells = self.get_empty_cells()
        if empty_cells:
            i, j = random.choice(empty_cells)
            value = 1 if random.random() < 0.9 else 2  # Representing 2 or 4 as
            # exponents
            self.set_tile(i, j, value)
            if value == 2:
                self.score -= 4
        else:
            self.over = True

    def move(self, direction: Direction) -> bool:
        """
        Move the tiles in the given direction.

        Args:
            direction (Direction): The direction in which to move the tiles.

        Returns:
            bool: True if the tiles were moved, False otherwise.
        """
        previous_board = self.board_int

        if direction == Board.Direction.UP:
            self.board_int = self.execute_move(0, self.board_int)
        elif direction == Board.Direction.DOWN:
            self.board_int = self.execute_move(1, self.board_int)
        elif direction == Board.Direction.LEFT:
            self.board_int = self.execute_move(2, self.board_int)
        elif direction == Board.Direction.RIGHT:
            self.board_int = self.execute_move(3, self.board_int)

        if self.board_int != previous_board:
            self.score = Board.score_board(self.board_int)
            self.place_random_tile()
            self.save_highscore()
            return True
        return False

    def get_valid_moves(self) -> list[Direction]:
        """
        Get a list of valid moves that can be performed on the board.

        :return: A list of valid directions.
        """
        valid_moves = []
        for direction in Board.Direction:
            new_board = self.board_int
            if direction == Board.Direction.UP:
                new_board = self.execute_move(0, self.board_int)
            elif direction == Board.Direction.DOWN:
                new_board = self.execute_move(1, self.board_int)
            elif direction == Board.Direction.LEFT:
                new_board = self.execute_move(2, self.board_int)
            elif direction == Board.Direction.RIGHT:
                new_board = self.execute_move(3, self.board_int)
            if new_board != self.board_int:
                valid_moves.append(direction)
        return valid_moves

    def get_empty_cells(self) -> list[tuple[int, int]]:
        """
        Get the empty cells on the board.

        :return: A list of empty cells.
        """
        empty_cells = []
        for i in range(4):
            for j in range(4):
                if self.get_tile(i, j, true_value=True) == 0:
                    empty_cells.append((i, j))
        return empty_cells

    def get_highest_tile(self) -> int:
        """
        Get the value of the highest tile on the board.

        :return: The value of the highest tile.
        """
        max_rank = 0
        temp_board = self.board_int
        while temp_board != 0:
            rank = temp_board & 0xF
            if rank > max_rank:
                max_rank = rank
            temp_board >>= 4
        return 2 ** max_rank if max_rank > 0 else 0

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        :return: True if the game is over, False otherwise.
        """
        if self.get_empty_cells():
            return False
        for move in range(4):
            if self.execute_move(move, self.board_int) != self.board_int:
                return False
        self.over = True
        return True

    def __str__(self) -> str:
        """Return a string representation of the board."""
        lines = []
        for i in range(4):
            line = []
            for j in range(4):
                rank = self.get_tile(i, j)
                value = (1 << rank) if rank != 0 else 0
                line.append(str(value).rjust(5))
            lines.append(" ".join(line))
        return "\n".join(lines)

    # Precomputed tables
    ROW_MASK = 0xFFFF
    COL_MASK = 0x000F000F000F000F

    # Initialize move tables and heuristic tables
    row_left_table = {}
    row_right_table = {}
    col_up_table = {}
    col_down_table = {}
    heuristic_score_table = {}
    score_table = {}

    @classmethod
    def init_tables(cls):
        """Initialize the precomputed tables."""
        loss_penalty = -199999.53225143452
        weight_empty_cells = 13212.487009009097
        weight_merges = 10958.238196752995
        weight_monotonicity = 5487.9254463412135
        weight_sum_values = 7053.376118104525
        power_monotonicity = 1.526531723375204
        power_sum_values = 0.3395712983957447

        for row in range(65536):
            line = [
                (row >> 0) & 0xF,
                (row >> 4) & 0xF,
                (row >> 8) & 0xF,
                (row >> 12) & 0xF
            ]

            # Calculate score and heuristic score
            score = 0
            for i in range(4):
                rank = line[i]
                if rank >= 2:
                    score += (1 << rank) * (rank - 1)
            cls.score_table[row] = score

            empty = line.count(0)
            merges = 0
            prev_rank = 0
            counter = 0
            sum_values = 0.0
            for i in range(4):
                rank = line[i]
                sum_values += pow(rank, power_sum_values)
                if rank == 0:
                    continue
                if prev_rank == rank:
                    counter += 1
                elif counter > 0:
                    merges += 1 + counter
                    counter = 0
                prev_rank = rank
            if counter > 0:
                merges += 1 + counter

            monotonicity_left = 0
            monotonicity_right = 0
            for i in range(1, 4):
                if line[i - 1] > line[i]:
                    monotonicity_left += pow(line[i - 1], power_monotonicity) - pow(
                        line[i], power_monotonicity)
                else:
                    monotonicity_right += pow(line[i], power_monotonicity) - pow(
                        line[i - 1], power_monotonicity)

            heuristic_score = loss_penalty + \
                              weight_empty_cells * empty + \
                              weight_merges * merges - \
                              weight_monotonicity * min(monotonicity_left,
                                                        monotonicity_right) - \
                              weight_sum_values * sum_values

            cls.heuristic_score_table[row] = heuristic_score

            # Execute a move to the left
            results = cls.execute_row_move(line)
            result = results['result']
            # Pack the result into a 16-bit integer
            packed_result = (result[0] << 0) | (result[1] << 4) | (result[2] << 8) | (
                    result[3] << 12)

            # Store the difference between the original row and the result
            cls.row_left_table[row] = row ^ packed_result

            # Reverse the row for the right move
            rev_row = cls.reverse_row(row)
            rev_result = cls.reverse_row(packed_result)
            cls.row_right_table[rev_row] = rev_row ^ rev_result

            # For columns, store the difference between the original column and the
            # result
            cls.col_up_table[row] = cls.unpack_col(row) ^ cls.unpack_col(packed_result)
            cls.col_down_table[rev_row] = cls.unpack_col(rev_row) ^ cls.unpack_col(
                rev_result)

    @staticmethod
    def reverse_row(row: int) -> int:
        """
        Reverse the row (16 bits).

        :param row: The row to reverse.
        :return: The reversed row.
        """
        return ((row >> 12) & 0xF) | ((row >> 4) & 0xF0) | ((row << 4) & 0xF00) | (
                (row << 12) & 0xF000)

    @staticmethod
    def unpack_col(row: int) -> int:
        """
        Unpack a row into a column representation.

        :param row: The row to unpack.
        :return: The unpacked column.
        """
        tmp = row
        return (tmp & 0xF) << 0 | \
            (tmp & 0xF0) << 12 | \
            (tmp & 0xF00) << 24 | \
            (tmp & 0xF000) << 36

    @staticmethod
    def execute_row_move(line: list[int]) -> dict:
        """
        Execute a move to the left on a row.

        :param line: The row to move.
        :return: The result of the move.
        """
        result: list[int] = []
        score = 0
        # Step 1: Slide tiles to the left
        tiles = [tile for tile in line if tile != 0]

        # Step 2: Merge tiles
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                # Merge tiles
                merged_value = tiles[i] + 1 if tiles[i] < 15 else tiles[i]
                result.append(merged_value)
                score += (1 << merged_value)
                i += 2  # Skip the next tile since it's merged
            else:
                result.append(tiles[i])
                i += 1

        # Step 3: Fill the rest with zeros
        result.extend([0] * (4 - len(result)))

        return {'result': result, 'score': score}

    @staticmethod
    def execute_move(move: int, board_int: int) -> int:
        """
        Execute a move in the given direction.

        :param move: The direction of the move.
        :param board_int: The board state.
        :return: The new board state.
        """
        if move == 0:  # Up
            t = Board.transpose(board_int)
            t = Board.execute_row_move_full(t, Board.row_left_table)
            ret = Board.transpose(t)
        elif move == 1:  # Down
            t = Board.transpose(board_int)
            t = Board.execute_row_move_full(t, Board.row_right_table)
            ret = Board.transpose(t)
        elif move == 2:  # Left
            ret = Board.execute_row_move_full(board_int, Board.row_left_table)
        elif move == 3:  # Right
            ret = Board.execute_row_move_full(board_int, Board.row_right_table)
        else:
            raise ValueError("Invalid move")
        return ret

    @staticmethod
    def execute_row_move_full(board_int: int, table: dict[int, int]) -> int:
        """
        Execute a row move using the precomputed table.

        :param board_int: The board state.
        :param table: The precomputed table.
        :return: The new board state.
        """
        ret = board_int
        ret ^= int(table[(board_int >> 0) & 0xFFFF]) << 0
        ret ^= int(table[(board_int >> 16) & 0xFFFF]) << 16
        ret ^= int(table[(board_int >> 32) & 0xFFFF]) << 32
        ret ^= int(table[(board_int >> 48) & 0xFFFF]) << 48
        return ret

    @staticmethod
    def transpose(board_int: int) -> int:
        """
        Transpose the board.

        :param board_int: The board state.
        :return: The transposed board state.
        """
        a1 = board_int & 0xF0F00F0FF0F00F0F
        a2 = board_int & 0x0000F0F00000F0F0
        a3 = board_int & 0x0F0F00000F0F0000
        a = a1 | (a2 << 12) | (a3 >> 12)
        b1 = a & 0xFF00FF0000FF00FF
        b2 = a & 0x00FF00FF00000000
        b3 = a & 0x00000000FF00FF00
        return b1 | (b2 >> 24) | (b3 << 24)

    @classmethod
    def score_board(cls, board_int: int) -> int:
        """
        Calculate the actual score of the board.

        :param board_int: The board state.
        :return: The score of the board.
        """
        return cls.score_table[(board_int >> 0) & cls.ROW_MASK] + \
            cls.score_table[(board_int >> 16) & cls.ROW_MASK] + \
            cls.score_table[(board_int >> 32) & cls.ROW_MASK] + \
            cls.score_table[(board_int >> 48) & cls.ROW_MASK]

    @staticmethod
    def get_empty_cells_from_board(board_int: int) -> list[tuple[int, int]]:
        """
        Get the empty cells on the board.

        :param board_int: The board state.
        :return: A list of empty cells
        """
        empty_cells = []
        for i in range(4):
            for j in range(4):
                index = Board.get_index(i, j)
                rank = (board_int >> index) & 0xF
                if rank == 0:
                    empty_cells.append((i, j))
        return empty_cells

    @staticmethod
    def is_game_over_static(board_int: int) -> bool:
        """Check if the game is over for the given board state."""
        if Board.get_empty_cells_from_board(board_int):
            return False
        for move in range(4):
            new_board_int = Board.execute_move(move, board_int)
            if new_board_int != board_int:
                return False
        return True


# Initialize the tables when the module is imported
Board.init_tables()
