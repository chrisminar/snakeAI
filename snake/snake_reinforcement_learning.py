"""Snake for reinforcement learning."""
from typing import Callable, List, Tuple

import numpy as np
from helper import Timer
from numpy import typing as npt

from snake import Snake


class SnakeRL(Snake):
    def __init__(self, *args, **kwargs) -> None:
        super(SnakeRL, self).__init__(
            kwargs['sizeX'], kwargs['sizeY'])  # expect sizeX, sizeY
        self.neural_net = kwargs['nn']
        self.food_x, self.food_y = self.spawn_food()
        self.grid[self.food_x][self.food_y] = -2  # set food on grid
        self.state_list = []
        self.move_list = []
        self.head_list = []

    def run_step(self, direction: str) -> None:
        if not self.gameover:
            self.moves += 1
            self.moves_since_food += 1
            if direction == 3:  # left
                self.run_single(-1, 0)
            elif direction == 1:  # right
                self.run_single(1, 0)
            elif direction == 2:  # down
                self.run_single(0, -1)
            elif direction == 0:  # up
                self.run_single(0, 1)
            else:  # invalid direction = no input
                self.run_single(self.Xdir, self.Ydir)

    def play(self, grid_func: Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]) -> None:
        while not self.gameover:
            new_dir, move, head_view = self.evaluate_next_step(grid_func)
            self.move_list.append(move)
            self.state_list.append(np.copy(self.grid))
            self.head_list.append(head_view)
            self.run_step(new_dir)
        return self.score

    def evaluate_next_step(self, grid_func: Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]) -> Tuple[int, List[int], npt.NDArray[np.int32]]:
        pre_processed_grid = grid_func(self.grid)  # preprocess grid
        head_view = SnakeRL.convert_head(
            self.X, self.Y, self.sizeX, self.sizeY, self.grid)
        policy = self.neural_net.evaluate(pre_processed_grid, head_view)

        out = [0, 0, 0, 0]
        new_dir = np.argmax(policy).astype(int)

        # check if nn direction is dead
        # if headView[newDir] == 0: #current trajotory is death
        #  if np.sum(headView) > 0: #at least one direction is free
        #    validIndex = np.where(headView)[0]
        #    newDir = np.random.choice(validIndex)
        #  else:
        #    pass#no directions are free, death is immenent
        # else:
        #  pass#current trajectory is ok

        out[new_dir] = 1

        return new_dir, out, head_view

    # look at head, return a boolean array [up, right, down, left] 0 means not ok to move, and 1 means ok to move
    @staticmethod
    def convert_head(x: int, y: int, grid_size_x: int, grid_size_y: int, grid: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        is_free = np.zeros((4,))

        if x == 0:  # on left wall
            pass
        elif grid[x-1, y] < 0:  # is left empty or food
            is_free[3] = 1

        if x == grid_size_x-1:  # on right wall wall
            pass
        elif grid[x+1, y] < 0:  # is right empty or food
            is_free[1] = 1

        if y == 0:  # on bot wall
            pass
        elif grid[x, y-1] < 0:  # is below empty or food
            is_free[2] = 1

        if y == grid_size_y-1:  # on top wall
            pass
        elif grid[x, y+1] < 0:  # is above empty or food
            is_free[0] = 1

        return is_free
