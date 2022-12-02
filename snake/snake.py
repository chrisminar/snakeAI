"""Snake game base class."""

import random
from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import Any, Optional, Tuple, Union

import numpy as np

from training.helper import (GRID_X, GRID_Y, SCORE_FOR_GAME_WIN,
                             SCORE_PENALTY_FOR_FAILURE, SCORE_PER_FOOD,
                             SCORE_PER_MOVE)


class GridEnum(IntEnum):
    """Grid cell meanings."""
    FOOD = -2  # grid value that represents food
    EMPTY = -1  # grid value that represents an empty space
    HEAD = 0  # grid value that represents the head
    BODY = 1  # grid value that represents the body


class Direction(IntEnum):
    """2d direction enumerators."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Snake(metaclass=ABCMeta):
    """Snake base class."""

    def __init__(self, x_grid_size: int = GRID_X, y_grid_size: int = GRID_Y) -> None:
        """Initialze snake game.

        Args:
            x_grid_size (int, optional): grid size in x direction. Defaults to GRID_X.
            y_grid_size (int, optional): grid size in y direction. Defaults to GRID_Y.
        """
        self.rng = np.random.default_rng()

        # initialize grid
        self.grid_size_x = x_grid_size  # width of grid
        self.grid_size_y = y_grid_size  # height of grid
        self.grid = np.full(
            (self.grid_size_x, self.grid_size_y), GridEnum.EMPTY.value, dtype=np.int32)
        self.grid_size = x_grid_size * y_grid_size

        # initialize snake
        self.head_x, self.head_y = self.rng.choice(
            np.argwhere(self.grid == GridEnum.EMPTY.value))
        self.length = 0  # current length of snake
        # set snake head on the grid
        self.grid[self.head_x, self.head_y] = GridEnum.HEAD.value

        # scoring
        self.score = 0  # current score
        self.moves = 0  # total moves taken
        self.moves_since_food = 0  # moves since food was last eaten

        # input
        direction = random.choice(list(Direction))
        self.direction_x, self.direction_y = self.direction_to_tuple(direction)

        # gamestate
        self.game_over = False

        # initialize food
        self.food_x, self.food_y = self.spawn_food()
        # set food on grid
        self.grid[self.food_x, self.food_y] = GridEnum.FOOD.value

    @abstractmethod
    def direction_to_tuple(self, direction: Union[Direction, Any]) -> Tuple[int, int]:
        """Convert direction to delta x and delta y.

        Args:
            direction (Direction): Direction to move head.

        Returns:
            Tuple[int,int]: Direction to move head in x and y.
        """
        raise NotImplementedError

    def run_single(self, x_direction: int, y_direction: int) -> None:
        """Run one step of snake game.

        Args:
            x_direction (int): Move head in this direction along the x axis.
            y_direction (int): Move head in this directino along the y axis.
        """
        self.direction_x = x_direction
        self.direction_y = y_direction
        self.step_time()

    def step_time(self) -> None:
        """Progress the game one time step."""
        # move head
        self.head_x += self.direction_x
        self.head_y += self.direction_y
        self.score += SCORE_PER_MOVE
        self.moves += 1

        # check if snake ate
        ate_this_turn = False
        if (self.head_x == self.food_x) and (self.head_y == self.food_y):
            self.length += 1
            self.food_x, self.food_y = self.spawn_food()
            self.grid[self.food_x, self.food_y] = GridEnum.FOOD.value
            self.score += SCORE_PER_FOOD
            ate_this_turn = True
            self.moves_since_food = 0
            if self.length == self.grid_size - 1:  # if snake is max length, the game has been won
                self.score += SCORE_FOR_GAME_WIN
                self.game_over = True
        else:
            self.moves_since_food += 1

        # move body
        self.grid[self.grid >= GridEnum.HEAD.value] += 1
        if not ate_this_turn:  # remove tail at end if the snake didn't grow
            self.grid[self.grid > self.length] = GridEnum.EMPTY.value

        # check if dead
        self.game_over = self.check_game_over()
        if self.game_over:
            self.score += SCORE_PENALTY_FOR_FAILURE
        else:
            self.grid[self.head_x, self.head_y] = GridEnum.HEAD.value

    def spawn_food(self) -> Tuple[int, int]:
        """Spawn food in an empty location.

        Returns:
            Tuple[int, int]: X and Y food location.
        """
        valid_food_spots = np.argwhere(self.grid == GridEnum.EMPTY.value)
        food_x, food_y = self.rng.choice(valid_food_spots)

        return food_x, food_y

    def check_game_over(self) -> bool:
        """Check if the game is over.

        Returns:
            bool: Is game over?
        """
        # check if we ran into a wall
        if self.head_x < 0:  # ran into left wall
            return True
        if self.head_x >= self.grid_size_x:  # ran into right wall
            return True
        if self.head_y < 0:  # upper wall
            return True
        if self.head_y >= self.grid_size_y:  # lower wall
            return True
        if self.grid[self.head_x, self.head_y] >= GridEnum.HEAD.value:  # head ran into body
            return True

        return False

    def _reset(self,
               *,
               head_x: int = 0,
               head_y: int = 0,
               food_x: Optional[int] = None,
               food_y: Optional[int] = None) -> None:
        """Reset the snake to the passed parameters.

        This is for testing purposes.

        Args:
            head_x (int): Head x position.
            head_y (int): Head y position.
            food_x (int): Food x position.
            food_y (int): Food y position.
        """
        if head_x == food_x and head_y == food_y:
            raise ValueError("Head and food can't be on same spot.")
        # reset grid
        self.grid.fill(GridEnum.EMPTY.value)

        # place head
        self.head_x = head_x
        self.head_y = head_y
        if not 0 <= self.head_x < self.grid_size_x or not 0 <= self.head_y < self.grid_size_y:
            raise ValueError("Invalid head position.")
        self.grid[self.head_x, self.head_y] = GridEnum.HEAD.value

        # place food
        if food_x is None:
            food_x = self.grid_size_x-1
        if food_y is None:
            food_y = self.grid_size_y-1
        self.food_x = food_x
        self.food_y = food_y
        if not 0 <= self.food_x < self.grid_size_x or not 0 <= self.food_y < self.grid_size_y:
            raise ValueError("Invalid food position")
        self.grid[food_x, food_y] = GridEnum.FOOD.value
