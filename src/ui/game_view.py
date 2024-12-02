from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, \
    QPushButton, QComboBox
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt, Slot, QTimer

from core.board import Board
from learn.rl_trainer import RLTrainer
from learn.random_strategy import RandomStrategy


class GameView(QWidget):
    """Game view for the 2048 game."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName("game_view")
        self.setFocus()  # Set focus to the game view to receive key events
        self.board = Board()
        self.trainer = RLTrainer(RandomStrategy())
        self.ai_timer = None

        self.game_over_label = None
        self.tile_widgets: list[list[QLabel]] = []
        self.highscore_label = QLabel(f"High Score: {self.board.load_highscore()}",
                                      self)
        self.score_label = QLabel(f"Score: {self.board.score}", self)

        self.parent_window = parent

        main_layout = QVBoxLayout(self)

        # Create title and buttons layout
        menu_layout = QVBoxLayout()
        menu_layout.addWidget(self.create_title())
        menu_layout.addLayout(self.create_menu_buttons())

        # Create scoreboard layout
        scoreboard_layout = QVBoxLayout()
        scoreboard_layout.addWidget(self.highscore_label)
        scoreboard_layout.addWidget(self.score_label)

        top_right_layout = QVBoxLayout()
        top_right_layout.addLayout(scoreboard_layout)
        top_right_layout.addLayout(self.create_ai_form())

        # Combine menu and scoreboard layouts
        top_layout = QHBoxLayout()
        top_layout.addLayout(menu_layout)
        top_layout.addLayout(top_right_layout)
        main_layout.addLayout(top_layout)

        self.grid_layout = self.create_game_board()
        main_layout.addLayout(self.grid_layout)

        # Render the initial board if not loaded
        loaded = self.board.load_state()
        if not loaded:
            pass  # Board already initializes with two random tiles
        self.update_board()

    @staticmethod
    def create_title() -> QLabel:
        """Create the title label."""
        result = QLabel("Magic 2048")
        result.setObjectName("title")
        result.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        return result

    def create_menu_buttons(self) -> QVBoxLayout:
        """Create the menu buttons layout."""
        result = QVBoxLayout()

        new_game_button = QPushButton("New Game")
        new_game_button.clicked.connect(self.new_game)
        result.addWidget(new_game_button)

        return_button = QPushButton("Return to Menu")
        return_button.clicked.connect(self.return_to_menu)
        result.addWidget(return_button)

        return result

    def create_ai_form(self) -> QVBoxLayout:
        """Create the AI form layout."""
        result = QVBoxLayout()

        model_selection = QComboBox()
        model_selection.addItem("Random")
        model_selection.addItem("Expectimax")
        # model_selection.addItem("DQN")
        model_selection.currentIndexChanged.connect(self.select_model)
        result.addWidget(model_selection)

        run_button = QPushButton("Run AI")
        run_button.clicked.connect(self.start_ai)
        train_button = QPushButton("Train AI")
        train_button.clicked.connect(lambda: print("Train AI"))

        action_layout = QHBoxLayout()
        action_layout.addWidget(run_button)
        action_layout.addWidget(train_button)
        result.addLayout(action_layout)

        return result

    @Slot(int)
    def select_model(self, index: int):
        """Select a model for the AI."""
        self.trainer.set_strategy(RLTrainer.Strategy(index))
        self.trainer.load()

    @Slot()
    def new_game(self):
        """Start a new game by replacing the central widget."""
        # Remove the game over label if it exists
        if self.game_over_label is not None:
            self.grid_layout.removeWidget(self.game_over_label)
            self.game_over_label.deleteLater()
            self.game_over_label = None

        # Reset the board and update the UI
        self.board = Board()
        self.update_board()
        self.update_score()

    @Slot()
    def start_ai(self):
        """Start the AI to play the game."""
        if self.board.is_game_over():
            self.new_game()
        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self.run_ai_step)
        self.ai_timer.start(50)

    @Slot()
    def run_ai_step(self):
        """Run the AI for one step."""
        if self.board.is_game_over():
            self.ai_timer.stop()
            return

        action = self.trainer.select_action(board=self.board)
        self.simulate_move(action)

    @Slot()
    def return_to_menu(self):
        """Return to the main menu."""
        if self.parent_window:
            if hasattr(self.parent_window, "central_widget"):
                self.parent_window.central_widget.deleteLater()
            if hasattr(self.parent_window, "setup_menu"):
                self.parent_window.setup_menu()

    def update_highscore(self):
        """Update the high score label."""
        self.highscore_label.setText(f"High Score: {self.board.load_highscore()}")

    def update_score(self):
        """Update the score label."""
        self.score_label.setText(f"Score: {self.board.score}")

    def create_game_board(self) -> QGridLayout:
        """Create the game board grid."""
        grid_layout = QGridLayout()
        grid_layout.setSpacing(5)

        # Create placeholders for tiles
        self.tile_widgets = [
            [QLabel("", self) for _ in range(self.board.size)] for _ in
            range(self.board.size)
        ]

        for row in range(self.board.size):
            for col in range(self.board.size):
                tile = self.tile_widgets[row][col]
                tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
                tile.setMinimumSize(100, 100)
                tile.setObjectName("tile")
                grid_layout.addWidget(tile, row, col)

        return grid_layout

    def update_board(self):
        """Update the board with the current state of the game."""
        for row in range(self.board.size):
            for col in range(self.board.size):
                value = self.board.get_tile(row, col, true_value=True)
                tile = self.tile_widgets[row][col]
                if value == 0:
                    tile.setText("")
                    tile.setObjectName("tile")
                else:
                    tile.setText(str(value))
                    tile.setObjectName(f"tile_{value}")
                # Force style refresh
                tile.style().unpolish(tile)
                tile.style().polish(tile)
                tile.update()

    def simulate_move(self, direction: Board.Direction):
        action_key_map = {
            Board.Direction.UP: Qt.Key.Key_Up,
            Board.Direction.DOWN: Qt.Key.Key_Down,
            Board.Direction.LEFT: Qt.Key.Key_Left,
            Board.Direction.RIGHT: Qt.Key.Key_Right
        }
        self.keyPressEvent(
            QKeyEvent(QKeyEvent.Type.KeyPress, action_key_map[direction],
                      Qt.KeyboardModifier.NoModifier)
        )

    # pylint: disable=invalid-name
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events for moving tiles."""
        key = event.key()
        moved = False
        if key == Qt.Key.Key_Up:
            moved = self.board.move(Board.Direction.UP)
        elif key == Qt.Key.Key_Down:
            moved = self.board.move(Board.Direction.DOWN)
        elif key == Qt.Key.Key_Left:
            moved = self.board.move(Board.Direction.LEFT)
        elif key == Qt.Key.Key_Right:
            moved = self.board.move(Board.Direction.RIGHT)
        else:
            super().keyPressEvent(event)
        if moved:
            self.update_board()
            self.update_score()
            self.update_highscore()
            if self.board.is_game_over():
                self.show_game_over()
                self.board.clear_state()
            else:
                self.board.save_state()

    def show_game_over(self):
        """Display a game over message."""
        if self.game_over_label is None:
            self.game_over_label = QLabel("Game Over", self)
            self.game_over_label.setObjectName("game_over")
            self.game_over_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid_layout.addWidget(self.game_over_label, 0, 0, 4, 4)

    def get_board_state(self):
        """Get the board state as a 2D list of tile values."""
        return [
            [self.board.get_tile(i, j, true_value=True) for j in range(self.board.size)]
            for i in
            range(self.board.size)]
