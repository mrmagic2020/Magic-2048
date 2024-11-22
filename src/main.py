import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow


def load_stylesheet(filepath: str) -> str:
    """Load QSS stylesheet from a file."""
    if getattr(sys, "frozen", False):  # If bundled with PyInstaller
        # noinspection PyUnresolvedReferences, PyProtectedMember
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent.parent

    stylesheet_path = base_path / filepath
    with open(stylesheet_path, "r", encoding="utf-8") as file:
        return file.read()


def main():
    """Entry point for the 2048 game application."""
    app = QApplication(sys.argv)

    # Apply stylesheet
    stylesheet_path = "resources/styles/main.qss"
    app.setStyleSheet(load_stylesheet(stylesheet_path))

    # Create and show the main window
    window = MainWindow()
    window.show()

    # Start the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
