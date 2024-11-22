import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.learn.rl_strategy import RLStrategy
from src.core.board import Board


class DQN(nn.Module):
    """
    A Deep Q-Network (DQN) model.

    :var fc1: The first fully connected layer.
    :var fc2: The second fully connected layer.
    :var fc3: The third fully connected layer.
    """

    def __init__(self, input_size, output_size):
        """
        Initializes the DQN model with the given input and output sizes.

        :param input_size: The size of the input tensor.
        :param output_size: The size of the output tensor.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the DQN model.

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """
    A class used to represent a Replay Buffer for storing experiences.

    :var buffer: The buffer for storing experiences.
    """

    @dataclass
    class Experience:
        """
        A data class representing an experience in the replay buffer.

        :var state: The current state.
        :var action: The action taken.
        :var reward: The reward received.
        :var next_state: The next state.
        :var done: Whether the episode is done.
        """
        state: list[list[int]]
        action: int
        reward: float
        next_state: list[list[int]]
        done: bool

    def __init__(self, capacity: int):
        """
        Initializes the ReplayBuffer with a given capacity.

        :param capacity: The maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Experience):
        """
        Adds an experience to the buffer.

        :param experience: The experience to add.
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """
        Samples a batch of experiences from the buffer.

        :param batch_size: The number of experiences to sample.
        :return: A list of sampled experiences.
        """
        return random.sample(self.buffer, batch_size)

    @property
    def size(self) -> int:
        """
        Returns the current size of the buffer.

        :return: The size of the buffer.
        """
        return len(self.buffer)


class DQNStrategy(RLStrategy):
    """
    Deep Q-Learning (DQN) strategy for reinforcement learning.

    :var model: The DQN model.
    :var target_model: The target DQN model.
    :var replay_buffer: The replay buffer.
    :var optimizer: The optimizer for training the model.
    :var batch_size: The batch size for training the model.
    :var gamma: The discount factor for future rewards.
    :var epsilon: The exploration rate.
    :var epsilon_decay: The decay rate for the exploration rate.
    :var epsilon_min: The minimum exploration rate.
    """

    def __init__(self, state_size: int, action_size: int, buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 gamma: float = 0.99, lr: float = 0.001, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.1):
        super().__init__()
        self.model: DQN = DQN(state_size, action_size)
        self.target_model: DQN = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.replay_buffer: ReplayBuffer = ReplayBuffer(buffer_capacity)
        self.optimizer: optim.Adam = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size: int = batch_size
        self.gamma: float = gamma

        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min

        self.train_step_counter = 0
        self.target_update_freq = 1000

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def save_path(self) -> str:
        """
        The save path for the model.

        :return: The save path.
        """
        return str(Path.home() / ".2048-AI/dqn_model.pth")

    def save(self):
        """
        Save the model to a file.
        """
        if not Path(self.save_path).parent.exists():
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Saved model to {self.save_path}")

    def load(self):
        """
        Load the model from a file.
        """
        if Path(self.save_path).exists():
            self.model.load_state_dict(torch.load(self.save_path, weights_only=True))
            print(f"Loaded model from {self.save_path}")

    def train(self, board: Board, **kwargs):
        """
        Train the model using experiences from the replay buffer.
        :param board: The current board state.
        :param kwargs:
        """
        if self.replay_buffer.size < self.batch_size:
            return  # wait until enough experiences are collected

        # Sample a batch of experiences from the replay buffer
        batch: list[ReplayBuffer.Experience] = self.replay_buffer.sample(
            self.batch_size)

        # Extract the states, actions, rewards, next states, and done flags
        states = torch.tensor(
            [self.__flatten_state(experience.state) for experience in batch],
            dtype=torch.float32)
        actions = torch.tensor([experience.action for experience in batch],
                               dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([experience.reward for experience in batch],
                               dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(
            [self.__flatten_state(experience.next_state) for experience in batch],
            dtype=torch.float32)
        dones = torch.tensor([experience.done for experience in batch],
                             dtype=torch.float32).unsqueeze(1)

        # Compute current Q values
        q_values: torch.Tensor = self.model(states).gather(1, actions)

        # Compute target Q values
        next_q_values: torch.Tensor = self.target_model(next_states).max(dim=1)[
            0].unsqueeze(1)
        target_q_values: torch.Tensor = rewards + (
                1 - dones) * self.gamma * next_q_values

        # Compute loss and update the model
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # print(f"Loss: {loss.item()}, Epsilon: {self.epsilon}")

        # Update the target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()

    def select_action(self, board: Board) -> Board.Direction:
        """
        Select an action based on the current state.

        :param board: The current board state.
        :return: The selected action.
        """
        state = self.__board_to_state(board)
        valid_moves = board.get_valid_moves()

        if random.random() < self.epsilon:
            # Exploration: select a random action
            return random.choice(valid_moves)
        else:
            # Exploitation: select the best action based on Q-values
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state_tensor)

            # Map Q-values to valid moves
            valid_move_indices = [list(Board.Direction).index(move) for move in
                                  valid_moves]
            q_values = q_values[0, valid_move_indices]

            best_action_idx = torch.argmax(q_values).item()
            # print(f"best_action_idx: {best_action_idx}")
            return valid_moves[best_action_idx]

    def update_target_network(self):
        """
        Update the target network by copying the weights from the model.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    @staticmethod
    def calculate_reward(prev_board: Board, next_board: Board, reward: float) -> float:
        """
        Calculate the reward for the given transition.

        :param prev_board: The previous board state.
        :param next_board: The next board state.
        :param reward: The reward for the transition.
        :return: The reward for the transition.
        """
        merged_value = sum(sum(next_board.tiles, [])) - sum(sum(prev_board.tiles, []))
        if merged_value > 0:
            reward += math.log2(merged_value)

        if next_board.is_game_over():
            reward -= 1000

        # Encourage freeing up space
        prev_empty_cells = len(prev_board.get_empty_cells())
        next_empty_cells = len(next_board.get_empty_cells())
        if next_empty_cells > prev_empty_cells:
            reward += (next_empty_cells - prev_empty_cells) * 10

        # Encourage monotonicity in rows and columns
        monotonicity = 0
        for row in next_board.tiles:
            for i in range(next_board.size - 1):
                if row[i] <= row[i + 1]:
                    monotonicity += 1
        for col in zip(*next_board.tiles):
            for i in range(next_board.size - 1):
                if col[i] <= col[i + 1]:
                    monotonicity += 1
        reward += monotonicity

        corner_bonus = 0
        if next_board.tiles[0][0] > 0:
            corner_bonus += next_board.tiles[0][0]
        if next_board.tiles[0][-1] > 0:
            corner_bonus += next_board.tiles[0][-1]
        if next_board.tiles[-1][0] > 0:
            corner_bonus += next_board.tiles[-1][0]
        if next_board.tiles[-1][-1] > 0:
            corner_bonus += next_board.tiles[-1][-1]
        reward += corner_bonus * 10

        return reward

    @staticmethod
    def __board_to_state(board: Board) -> list[float]:
        """
        Convert the board state to a tensor for the model input.

        :param board: The current board state.
        :return: The tensor representing the board state.
        """
        return [0 if tile == 0 else math.log2(tile) for row in board.tiles for tile in
                row]

    @staticmethod
    def __flatten_state(state: list[list[int]]) -> list[float]:
        """
        Flatten the 2D board state into a 1D list.
        :param state: The 2D board state.
        :return: A flattened 1D list representing the board state.
        """
        return [0 if tile == 0 else math.log2(tile) for row in state for tile in row]
