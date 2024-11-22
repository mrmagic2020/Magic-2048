from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, \
    QMenu
from PySide6.QtCore import Qt, Slot, QKeyCombination
from src.ui.game_view import GameView


class MainWindow(QMainWindow):
    """Main window for the 2048 game."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Magic 2048")
        self.setGeometry(100, 100, 800, 600)

        menu_bar = self.menuBar()

        file_menu = QMenu("File", self)
        file_menu.addAction("New Game", QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_N),
                            self.new_game)
        file_menu.addAction("Load Game", lambda: print("Load Game"))
        menu_bar.addMenu(file_menu)

        window_menu = QMenu("Window", self)
        window_menu.addAction("Show Menu", QKeyCombination(Qt.Key.Key_Escape),
                              self.setup_menu)
        menu_bar.addMenu(window_menu)
        self.setMenuBar(menu_bar)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.menu_layout = QVBoxLayout(self.central_widget)

        self.setup_menu()

    def setup_menu(self):
        """Set up the menu layout."""
        # Clear the central widget if it already exists
        if self.central_widget:
            self.central_widget.deleteLater()

        # Create a new central widget and layout for the menu
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.menu_layout = QVBoxLayout(self.central_widget)

        self.menu_layout.addLayout(self.create_title())
        self.menu_layout.addWidget(self.create_menu_button("New Game", self.new_game))
        self.menu_layout.addWidget(
            self.create_menu_button("Load Game", lambda: print("Load Game")))
        self.menu_layout.addWidget(self.create_menu_button("Quit", self.close))

    @staticmethod
    def create_title() -> QVBoxLayout:
        """Create the title layout."""
        result = QVBoxLayout()
        result.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
        title = QLabel("Magic 2048")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result.addWidget(title)
        return result

    @staticmethod
    def create_menu_button(text: str, clicked: callable) -> QPushButton:
        """Create a menu button with the given text and clicked callback."""
        result = QPushButton(text)
        result.setObjectName("menu_button")
        result.clicked.connect(clicked)
        return result

    @Slot()
    def new_game(self):
        """Start a new game by replacing the central widget."""
        self.central_widget.deleteLater()  # Delete the current central widget
        self.central_widget = GameView(self)  # Create a new game view
        self.setCentralWidget(self.central_widget)
