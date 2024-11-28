import copy
import json
import pickle
from multiprocessing import Pool

import cma
import argparse
import numpy as np

from src.learn.rl_strategy import RLStrategy
from src.core.board import Board


class Expectimax(RLStrategy):
    """Expectimax strategy optimized for bitboard representation."""

    def __init__(self, depth: int = 5, params=None):
        """
        Initialize the expectimax strategy.

        :param depth: The depth of the expectimax search tree.
        """
        self.depth: int = depth
        self.transposition_table = {}
        if params is None:
            self.params = {
                'base_constant': -200000.0,
                'weight_empty_cells': 370.0,
                'weight_merges': 800.0,
                'weight_monotonicity': 247.0,
                'weight_sum_values': 11.0,
                'power_monotonicity': 4.0,
                'power_sum_values': 3.5,
                'weight_corner_bonus': 10000.0,
            }
        else:
            self.params = params

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
                score = self.__expectimax(new_board.board_int, self.depth,
                                          is_maximizing=False)
                if score > best_score:
                    best_score = score
                    best_action = action

        board.score = Board.score_board(board.board_int)
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
                index = Board.get_bit_index(pos[0], pos[1])
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

    def __evaluate_board(self, board_int: int) -> float:
        params = self.params
        grid = self.extract_grid(board_int)

        # Compute heuristic features
        empty = sum(tile == 0 for row in grid for tile in row)
        sum_values = sum(
            pow(tile, params['power_sum_values']) for row in grid for tile in row if
            tile != 0)

        # Merges and monotonicity (computed over rows)
        merges = 0
        monotonicity = 0
        for row in grid:
            merges += self.compute_merges(row)
            monotonicity += self.compute_monotonicity(row, params['power_monotonicity'])

        # Heuristic score
        score = params['base_constant'] + \
                params['weight_empty_cells'] * empty + \
                params['weight_merges'] * merges - \
                params['weight_monotonicity'] * monotonicity - \
                params['weight_sum_values'] * sum_values

        # Corner bonus/penalty
        max_rank = max(max(row) for row in grid)
        corner_ranks = [grid[0][0], grid[0][3], grid[3][0], grid[3][3]]
        if max_rank in corner_ranks:
            score += params['weight_corner_bonus'] * max_rank
        else:
            score -= params['weight_corner_bonus'] * max_rank

        return score

    # Helper methods for extracting grid, computing merges, and monotonicity
    @staticmethod
    def extract_grid(board_int):
        grid = [[0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                index = Board.get_bit_index(i, j)
                rank = (board_int >> index) & 0xF
                grid[i][j] = rank
        return grid

    @staticmethod
    def compute_merges(line):
        merges = 0
        prev_tile = None
        for tile in line:
            if tile != 0:
                if tile == prev_tile:
                    merges += 1
                prev_tile = tile
            else:
                prev_tile = None
        return merges

    @staticmethod
    def compute_monotonicity(line, power):
        mono_left = sum(
            pow(line[i], power) - pow(line[i + 1], power) for i in range(3) if
            line[i] > line[i + 1])
        mono_right = sum(
            pow(line[i + 1], power) - pow(line[i], power) for i in range(3) if
            line[i] < line[i + 1])
        return min(mono_left, mono_right)


def play_game(board: Board, expectimax: Expectimax) -> int:
    """
    Play a game using the expectimax strategy.
    :param board: The game board.
    :param expectimax: The expectimax strategy.
    :return: The final score of the game.
    """
    while not board.is_game_over():
        action: Board.Direction = expectimax.select_action(board)
        if action is None:
            break
        board.move(action)
    return board.score


def objective_function(param_vector: list, games: int, depth: int) -> float:
    """
    Objective function to optimize the expectimax strategy
    :param param_vector: The parameter vector.
    :param games: The number of games to play for each solution.
    :param depth: The depth of the expectimax search tree.
    :return: The fitness value.
    """
    params = {
        'base_constant': param_vector[0],
        'weight_empty_cells': param_vector[1],
        'weight_merges': param_vector[2],
        'weight_monotonicity': param_vector[3],
        'weight_sum_values': param_vector[4],
        'power_monotonicity': param_vector[5],
        'power_sum_values': param_vector[6],
        'weight_corner_bonus': param_vector[7],
    }
    total_score = 0
    for _ in range(games):
        board = Board()
        expectimax = Expectimax(depth=depth, params=params)
        game_score = play_game(board, expectimax)
        total_score += game_score
    average_score = total_score / games
    return -average_score


def objective_function_parallel(solutions_: list, processes: int = 7,
                                games: int = 5, depth: int = 5) -> list:
    """
    Evaluate the objective function in parallel.
    :param solutions_: The list of solutions to evaluate.
    :param processes: The number of processes to use.
    :param games: The number of games to play for each solution.
    :param depth: The depth of the expectimax search tree.
    :return: The list of fitness values.
    """
    with Pool(processes=processes) as pool:
        args_ = [(solution, games, depth) for solution in solutions_]
        fitness_ = pool.starmap(objective_function, args_)
    return fitness_


def save_progress(filename: str, es_: cma.CMAEvolutionStrategy):
    """
    Save the progress of the CMA-ES optimisation to a file.
    :param filename: The filename to save the progress to.
    :param es_: The CMA-ES object.
    :return: True if the progress was saved successfully, False otherwise.
    """
    s = es_.pickle_dumps()
    try:
        with open(filename, "wb") as file:
            file.write(s)
    except Exception as e:
        print(f"Error saving progress: {e}")
        return False
    return True


def load_progress(filename: str) -> cma.CMAEvolutionStrategy or None:
    """
    Load the progress of the CMA-ES optimisation from a file.
    :param filename: The filename to load the progress from.
    :return: The CMA-ES object or None if an error occurred.
    """
    try:
        with open(filename, "rb") as file:
            s = file.read()
    except Exception as e:
        print(f"Error loading progress: {e}")
        return None
    return pickle.loads(s)


def save_params(filename: str, params: np.ndarray) -> bool:
    """
    Save the parameters of the expectimax strategy to a file as a JSON object.
    :param filename: The filename to save the parameters to.
    :param params: The parameters to save.
    :return: True if the parameters were saved successfully, False otherwise.
    """
    # Convert numpy arrays to lists
    params = params.tolist()
    # Build the JSON object
    params = {
        'base_constant': params[0],
        'weight_empty_cells': params[1],
        'weight_merges': params[2],
        'weight_monotonicity': params[3],
        'weight_sum_values': params[4],
        'power_monotonicity': params[5],
        'power_sum_values': params[6],
        'weight_corner_bonus': params[7],
    }
    try:
        with open(filename, "w") as file:
            # noinspection PyTypeChecker
            json.dump(params, file)
    except Exception as e:
        print(f"Error saving parameters: {e}")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="optimisation.py",
        description="Optimise the Expectimax strategy for 2048."
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the progress file.")
    parser.add_argument("filename", nargs="?", default="expectimax_progress.pkl",
                        help="The filename to save the progress to.")
    parser.add_argument("--processes", type=int, default=7,
                        help="Number of processes to use.")
    parser.add_argument("--games", type=int, default=5,
                        help="Number of games to play for each solution.")
    parser.add_argument("--depth", type=int, default=5,
                        help="Depth of the expectimax search tree.")
    args = parser.parse_args()

    print(f"Optimising Expectimax strategy with CMA-ES (games={args.games}, "
          f"processes={args.processes}, depth={args.depth})...")

    initial_param_vector = [
        -197338.48676143715,  # base_constant
        13212.487009009097,  # weight_empty_cells
        10958.238196752995,  # weight_merges
        5487.9254463412135,  # weight_monotonicity
        7053.376118104525,  # weight_sum_values
        1.526531723375204,  # power_monotonicity
        0.3395712983957447,  # power_sum_values
        18875.884099326922,  # weight_corner_bonus
    ]

    bounds = [
        [-1e6, 0, 0, 0, 0, 0.1, 0.1, 0],
        [0, 1e5, 1e5, 1e5, 1e5, 10, 10, 1e5],
    ]

    es = cma.CMAEvolutionStrategy(initial_param_vector, 5000.0, {'bounds': bounds})
    if args.resume:
        es = load_progress(args.filename)
        if es is None:
            print("Error loading progress from file. Starting from scratch...")
            es = cma.CMAEvolutionStrategy(initial_param_vector, 5000.0,
                                          {'bounds': bounds})

    try:
        while not es.stop():
            solutions = es.ask()
            fitness = objective_function_parallel(solutions, args.processes, args.games,
                                                  args.depth)
            es.tell(solutions, fitness)
            es.disp()
            save_progress(args.filename, es)
            save_params("expectimax_params.json", es.result.xbest)
    except KeyboardInterrupt:
        print("User interrupted the optimisation process.")
        print("Saving the best solution found so far...")
        save_progress(args.filename, es)

    if es.result.xbest is None:
        print("No best solution found.")
        exit(1)
    save_params("expectimax_params.json", es.result.xbest)

    best_solution = es.result.xbest
    best_fitness = -es.result.fbest

    print("Best parameters found:")
    print(f"Base Constant: {best_solution[0]}")
    print(f"Weight Empty Cells: {best_solution[1]}")
    print(f"Weight Merges: {best_solution[2]}")
    print(f"Weight Monotonicity: {best_solution[3]}")
    print(f"Weight Sum Values: {best_solution[4]}")
    print(f"Power Monotonicity: {best_solution[5]}")
    print(f"Power Sum Values: {best_solution[6]}")
    print(f"Weight Corner Bonus: {best_solution[7]}")
    print(f"Best Average Score: {best_fitness}")
