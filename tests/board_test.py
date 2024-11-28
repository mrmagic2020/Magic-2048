import json
from pathlib import Path

import pytest

from core.board import Board


class Helper:
    @staticmethod
    def setup():
        test_data_dir = Path('tests/test_data')
        if not test_data_dir.exists():
            test_data_dir.mkdir()
        Board.state_path = test_data_dir / 'board_state.txt'
        Board.highscore_path = test_data_dir / 'highscore.txt'

    @staticmethod
    def cleanup():
        Board.state_path.unlink(missing_ok=True)
        Board.highscore_path.unlink(missing_ok=True)


@pytest.fixture
def helpers():
    return Helper


@pytest.fixture(autouse=True)
def test_setup(helpers):
    helpers.setup()
    yield
    helpers.cleanup()


def test_load_highscore(helpers):
    board = Board()
    highscore = board.load_highscore()
    assert (highscore == 0)

    with open(Board.highscore_path, 'w') as file:
        file.write('2048')
    highscore = board.load_highscore()
    assert (highscore == 2048)

    helpers.cleanup()


def test_save_highscore(helpers):
    board = Board()
    board.score = 1024
    board.save_highscore()
    with open(Board.highscore_path, 'r') as file:
        highscore = int(file.read())
    assert (highscore == 1024)

    board.score = 512
    board.save_highscore()
    with open(Board.highscore_path, 'r') as file:
        highscore = int(file.read())
    assert (highscore == 1024)

    helpers.cleanup()


def test_load_state(helpers):
    board = Board()
    success = board.load_state()
    assert (success is False)

    with open(Board.state_path, 'w') as file:
        state = {
            "size": 4,
            "board": 0,
            "score": 0,
            "won": True,
            "over": False
        }
        # noinspection PyTypeChecker
        json.dump(state, file)
    success = board.load_state()
    assert (success is True)
    assert (board.size == 4)
    assert (board.board_int == 0)
    assert (board.score == 0)
    assert (board.won is True)
    assert (board.over is False)

    with open(Board.state_path, 'w') as file:
        state = {
            "size": -1,
            "board": 0,
            "score": 0,
            "won": True,
            "over": False
        }
        # noinspection PyTypeChecker
        json.dump(state, file)
    success = board.load_state()
    assert (success is False)

    helpers.cleanup()


def test_save_state(helpers):
    board = Board()
    board.size = 4
    board.board_int = 0
    board.score = 0
    board.won = True
    board.over = False
    board.save_state()
    with open(Board.state_path, 'r') as file:
        state = json.load(file)
    assert (state["size"] == 4)
    assert (state["board"] == 0)
    assert (state["score"] == 0)
    assert (state["won"] is True)
    assert (state["over"] is False)

    helpers.cleanup()


def test_clear_state(helpers):
    board = Board()
    with open(Board.state_path, 'w') as file:
        file.write('{"size": 4, "board": 0, "score": 0, "won": true, "over": false}')
    board.clear_state()
    assert (not Board.state_path.exists())


def test_get_bit_index():
    assert (Board.get_bit_index(0, 0) == 0)
    assert (Board.get_bit_index(0, 1) == 4)
    assert (Board.get_bit_index(1, 0) == 16)
    assert (Board.get_bit_index(1, 1) == 20)


def test_get_tile():
    board = Board()
    board.board_int = \
        0b0000_0000_0000_0000_0000_0000_0000_0000_1000_0111_0110_0101_0100_0011_0010_0001
    assert (board.get_tile(0, 0) == 1)
    assert (board.get_tile(0, 0, true_value=True) == 2)
    assert (board.get_tile(0, 1) == 2)
    assert (board.get_tile(0, 1, true_value=True) == 4)
    assert (board.get_tile(0, 2) == 3)
    assert (board.get_tile(0, 2, true_value=True) == 8)
    assert (board.get_tile(0, 3) == 4)
    assert (board.get_tile(0, 3, true_value=True) == 16)
    assert (board.get_tile(1, 0) == 5)
    assert (board.get_tile(1, 1) == 6)
    assert (board.get_tile(1, 2) == 7)
    assert (board.get_tile(1, 3) == 8)
    for i in range(2, 4):
        for j in range(4):
            assert (board.get_tile(i, j) == 0)
            assert (board.get_tile(i, j, true_value=True) == 0)


def test_set_tile():
    board = Board()
    board.board_int = 0  # clear board
    board.set_tile(0, 0, 1)
    board.set_tile(0, 1, 2)
    board.set_tile(0, 2, 3)
    board.set_tile(0, 3, 4)
    board.set_tile(1, 0, 5)
    board.set_tile(1, 1, 6)
    board.set_tile(1, 2, 7)
    board.set_tile(1, 3, 8)
    assert (
            board.board_int ==
            0b0000_0000_0000_0000_0000_0000_0000_0000_1000_0111_0110_0101_0100_0011_0010_0001)

    board.set_tile(0, 0, 0)
    assert (
            board.board_int ==
            0b0000_0000_0000_0000_0000_0000_0000_0000_1000_0111_0110_0101_0100_0011_0010_0000)


def test_place_random_tile():
    board = Board()
    board.board_int = 0  # clear board
    board.place_random_tile()
    assert (board.board_int != 0)
    assert (len(board.get_empty_cells()) == 15)
    board.place_random_tile()
    assert (len(board.get_empty_cells()) == 14)


def test_execute_move():
    board = Board()
    board.board_int = 0
    board.set_tile(0, 0, 2)
    board.board_int = board.execute_move(Board.Direction.RIGHT.value, board.board_int)
    assert (board.get_tile(0, 3) == 2)
    board.set_tile(0, 0, 2)
    board.board_int = board.execute_move(Board.Direction.LEFT.value, board.board_int)
    assert (board.get_tile(0, 0) == 3)
    board.set_tile(2, 0, 3)
    board.board_int = board.execute_move(Board.Direction.DOWN.value, board.board_int)
    assert (board.get_tile(3, 0) == 4)
