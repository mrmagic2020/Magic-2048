from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDialog, QProgressBar


class TrainingDialog(QDialog):
    """Dialog window to display training progress."""

    def __init__(self, parent: QWidget, total_episodes: int):
        super().__init__(parent)
        self.setWindowTitle("Training AI")
        self.setModal(True)
        self.total_episodes = total_episodes

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, total_episodes)
        self.progress_bar.setValue(0)

        # Labels to show stats
        self.info_label = QLabel("Training in progress...", self)
        self.average_score_label = QLabel("Average Score: 0", self)
        self.highest_score_label = QLabel("Highest Score: 0", self)
        self.highest_tile_label = QLabel("Highest Tile: 0", self)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.average_score_label)
        layout.addWidget(self.highest_score_label)
        layout.addWidget(self.highest_tile_label)
        self.setLayout(layout)

    def update_progress(self, episode: int, average_score: float, highest_score: int,
                        highest_tile: int):
        """Update the progress bar and stats."""
        self.progress_bar.setValue(episode)
        self.average_score_label.setText(f"Average Score: {average_score:.2f}")
        self.highest_score_label.setText(f"Highest Score: {highest_score}")
        self.highest_tile_label.setText(f"Highest Tile: {highest_tile}")
