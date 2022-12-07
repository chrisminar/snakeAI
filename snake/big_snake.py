"""Parallelizable snake."""

from typing import List, Optional, Tuple

import numpy as np
from numpy import typing as npt

from snake.snake import Direction, GridEnum
from training.helper import (EXPLORATORY_MOVE_FRACTION, GRID_X, GRID_Y,
                             NUM_SELF_PLAY_GAMES)
from training.neural_net import NeuralNetwork
from training.play_games import grid_2_nn


class ParSnake:
    """Parallel snake class."""

    def __init__(self, grid_size_x: int = GRID_X, grid_size_y: int = GRID_Y, exploratory: bool = False, num_games: int = NUM_SELF_PLAY_GAMES) -> None:
        """_summary_

        Args:
            grid_size_x (int, optional): _description_. Defaults to GRID_X.
            grid_size_y (int, optional): _description_. Defaults to GRID_Y.
            exploratory (bool, optional): _description_. Defaults to False.
        """
        self.exploratory = exploratory
        self.rng = np.random.default_rng()

        # initialize grids
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.grid = np.full((num_games, self.grid_size_y,
                            self.grid_size_x), GridEnum.EMPTY.value, dtype=np.int32)
        self.grid_size = self.grid_size_x*self.grid_size_y

        # initialize snakes
        valids = self.choose_valids(
            self.grid == GridEnum.EMPTY.value)
        self.heads_x, self.heads_y = valids[:, 1], valids[:, 0]
        self.lengths = np.zeros((num_games,), dtype=np.int32)
        self.grid[:, self.heads_y, self.heads_x] = GridEnum.HEAD.value

        # scoring
        self.scores = np.zeros_like(self.lengths)
        self.num_moves = np.zeros_like(self.lengths)
        self.moves_since_food = np.zeros_like(self.lengths)

        # input
        self.direction_xs = np.zeros_like(self.lengths)
        self.direction_ys = np.zeros_like(self.lengths)

        # gamestate
        self.games_over = False

        # initialize food
        valids = self.choose_valids(
            self.grid == GridEnum.EMPTY.value)
        self.food_xs, self.food_ys = valids[:, 1], valids[:, 0]
        self.grid[:, self.food_ys, self.food_xs] = GridEnum.FOOD.value

        # init neural network
        self.neural_net = NeuralNetwork()

        # init tracking
        self.states = List[npt.NDArray[np.int32]] = []
        self.moves = List[npt.NDArray[np.int32]] = []
        self.heads = List[npt.NDArray[np.bool8]] = []

    def choose_valids(self, array: npt.NDArray[np.bool8], backups: Optional[npt.NDArray[np.bool8]] = None) -> npt.NDArray[np.int32]:
        """Choose valid indices from a 3d array

        Args:
            array (npt.NDArray[np.bool8]): Array of size (m,n,k). Will choose a random valid element from (n,k) for each m

        Returns:
            npt.NDArray[np.int32]: random indicies (m,2)
        """
        if array.ndim < 2:
            raise ValueError("Requires at least 2 dimensional array.")

        indicies = np.full((array.shape[0], 2), 0, dtype=np.int32)

        if backups is not None:
            for m, (one_snake, backup) in enumerate(zip(array, backups)):
                indicies[m] = self.get_random_valid(one_snake, backup)
        else:
            for m, one_snake in enumerate(array):
                indicies[m] = self.get_random_valid(one_snake)

        return indicies

    def get_random_valid(self, one_snake: npt.NDArray[np.bool8], backup: Optional[npt.NDArray[np.bool8]] = None) -> npt.NDArray[np.int32]:
        """_summary_

        Args:
            one_snake (npt.NDArray[np.bool8]): Array of size (n,k) Will choose a random valid element.

        Returns:
            npt.NDArray[np.int32]: Indices of random valid element
        """
        try:
            return self.rng.choice(np.argwhere(one_snake))
        except ValueError:
            if backup is not None:
                try:
                    return self.rng.choice(np.argwhere(backup))
                except ValueError:
                    pass
        return np.array([0, 0], dtype=np.int32)

    def run_single():
        pass

    def step_time():
        pass

    def directions_to_tuples():
        pass

    def play(self) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.bool8], npt.NDArray[np.int32]]:
        """Play all snake games till completion."""
        while not self.games_over:
            new_directions, new_moves, new_heads = self.evaluate()
            self.states.append(self.grid.copy())
            self.moves.append(new_moves)
            self.heads.append(new_heads)
            # YOU ARE HERE
            self.run_single(*self.directions_to_tuples(new_directions))
        return np.concatenate(self.states), np.concatenate(self.moves), np.concatenate(self.heads), self.scores

    def evaluate(self):
        pre_processed_grid = grid_2_nn(self.grid)
        heads = self.convert_heads()
        policy = self.neural_net.evaluate(state=pre_processed_grid, head=heads)

        # account for invalid choices from NN
        invalid_choice = np.all(policy == 0, axis=1)
        policy[invalid_choice, :] = heads[invalid_choice, :]

        # check that no invalid choices were handled from head
        invalid_choice = np.all(policy == 0, axis=1)
        # force up when no valid options
        policy[invalid_choice, 0] = Direction.UP.value

        # random moves
        new_dir = np.argmax(policy, axis=1).astype(int)
        if self.exploratory:
            random_idx = np.random.rand(
                new_dir.shape) > EXPLORATORY_MOVE_FRACTION
            new_dir[random_idx] = self.choose_valids(
                np.logical_not(heads[random_idx]))

        # make an array of [up, right, left, down] for each move where the chosen direction is 1 and others are 0
        next_direction_array = np.zeros((new_dir.size, 4), dtype=np.int32)
        np.put_along_axis(next_direction_array, np.expand_dims(
            new_dir.copy(), axis=1), 1, axis=1)

        return new_dir, next_direction_array, heads

    def check_game_over():
        pass

    def convert_head(self):
        is_free = np.zeros((self.grid.shape[0], 4), dtype=np.bool8)

        # is left empty or food
        left_ok = self.heads_x > 0 and self.grid[self.heads_y,
                                                 self.heads_x-1] < GridEnum.HEAD.value
        is_free[left_ok, Direction.LEFT.value] = 1

        # is right empty or food
        right_ok = self.heads_x < self.grid_size_x - \
            1 and self.grid[self.heads_y, self.heads_x+1] < GridEnum.HEAD.value
        is_free[right_ok, Direction.RIGHT.value] = 1

        # is above empty or food
        up_ok = self.heads_y > 0 and self.grid[self.heads_y -
                                               1, self.heads_x] < GridEnum.HEAD.value
        is_free[up_ok, Direction.UP.value] = 1

        # is below empty or food
        down_ok = self.heads_y < self.grid_size_y - \
            1 and self.grid[self.heads_y+1, self.heads_x] < GridEnum.HEAD.value
        is_free[down_ok, Direction.DOWN.value] = 1

        return is_free
