"""Snake for reinforcement learning."""
from typing import Callable, List, Tuple

import numpy as np
from helper import MAXIMUM_MOVES_WITHOUT_EATING, MAXIMUM_TOTAL_MOVES
from neural_net import NeuralNetwork
from numpy import typing as npt

from snake.snake import Direction, GridEnum, Snake


class SnakeRL(Snake):
    """Reinforcement learning snake."""

    def __init__(self, neural_net: NeuralNetwork, **kwargs) -> None:
        super().__init__(**kwargs)
        self.neural_net = neural_net
        self.state_list: List[npt.NDArray[np.int32]] = []
        self.move_list: List[npt.NDArray[np.int32]] = []
        self.head_list: List[npt.NDArray[np.int32]] = []

    def direction_to_tuple(self, direction: Direction) -> Tuple[int, int]:
        """Convert direction to delta x and delta y.

        Args:
            direction (Direction): Direction to move head.

        Returns:
            Tuple[int,int]: Direction to move head in x and y.
        """
        if direction is Direction.UP:
            return 0, -1
        if direction is Direction.RIGHT:
            return 1, 0
        if direction is Direction.DOWN:
            return 0, -1
        return -1, 0

    def play(self, grid_func: Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]) -> int:
        """Play a game of snake.

        Args:
            grid_func (Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]): Function to preprocess grid cell values.

        Returns:
            int: Score
        """
        while not self.game_over:
            new_direction, move, head_view = self.evaluate_next_step(grid_func)
            self.move_list.append(move)
            self.state_list.append(np.copy(self.grid))
            self.head_list.append(head_view)
            self.run_single(*self.direction_to_tuple(new_direction))
        return self.score

    def evaluate_next_step(self, grid_func: Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]) -> Tuple[int, List[int], npt.NDArray[np.int32]]:
        """Convert game step to neural net inputs, then run neural network.

        Args:
            grid_func (Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]): Function to preprocess grid cell values.

        Returns:
            Tuple[int, List[int], npt.NDArray[np.int32]]: Next direction, next direction as array, valid head travel directions
        """
        pre_processed_grid = grid_func(self.grid)
        head_view = self.convert_head()
        policy = self.neural_net.evaluate(pre_processed_grid, head_view)

        out = [0, 0, 0, 0]
        new_dir = np.argmax(policy).astype(int)
        out[new_dir] = 1

        return Direction[new_dir], out, head_view

    def check_game_over(self) -> bool:
        """Check if the game is over.

        Returns:
            bool: Is game over?
        """
        if self.moves_since_food < MAXIMUM_MOVES_WITHOUT_EATING:  # prevent loops and stagnation
            return True
        if self.moves > MAXIMUM_TOTAL_MOVES:  # upper limit on number of moves
            return True

        return super().check_game_over()

    # look at head, return a boolean array [up, right, down, left] 0 means not ok to move, and 1 means ok to move
    def convert_head(self) -> npt.NDArray[np.bool8]:
        """Which directions can the head move and not die.

        Returns:
            npt.NDArray[np.bool8]: Array of if it's valid to move in a given direction.
        """
        is_free = np.zeros((4,), dtype=np.bool8)

        # is left empty or food
        if self.head_x > 0 and self.grid[self.head_x-1, self.head_y] < GridEnum.HEAD.value:
            is_free[Direction.LEFT.value] = 1

        # is right empty or food
        if self.head_x < self.grid_size_x and self.grid[self.head_x+1, self.head_y] < GridEnum.HEAD.value:
            is_free[Direction.RIGHT.value] = 1

        # is above empty or food
        if self.head_y > 0 and self.grid[self.head_x, self.head_y-1] < GridEnum.HEAD.value:
            is_free[Direction.UP.value] = 1

        # is below empty or food
        if self.head_y < self.grid_size_y and self.grid[self.head_x, self.head_y+1] < GridEnum.HEAD.value:
            is_free[Direction.DOWN.value] = 1

        return is_free
