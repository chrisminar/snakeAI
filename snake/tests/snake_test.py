"""Test functionality of snake base class."""

from typing import Any, Tuple, Union

import numpy as np
import pytest

from snake.snake import Direction, GridEnum, Snake
from snake.snake_reinforcement_learning import SnakeRL
from training.helper import GRID_X, GRID_Y
from training.neural_net import NeuralNetwork


class SnakeDummy(Snake):
    """Dummy child of Snake to test the functions of the base class."""

    def direction_to_tuple(self, direction: Union[Direction, Any]) -> Tuple[int, int]:
        """Dummy."""
        raise NotImplementedError


def test_init() -> None:
    """Test snake initialization."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    assert not snake.game_over
    assert snake.grid.shape == np.zeros((GRID_X, GRID_Y)).shape


@pytest.mark.parametrize("food", [True, False])
@pytest.mark.parametrize("head_x, head_y, x_dir, y_dir", [(0, 0, 1, 0),
                                                          (1, 0, -1, 0),
                                                          (0, 0, 0, 1),
                                                          (0, 1, 0, -1)])
def test_run_single_no_food(food: bool, head_x: int, head_y: int, x_dir: int, y_dir: int) -> None:
    """Run single step, do not eat."""
    snake = SnakeDummy(GRID_X, GRID_Y)

    # reset grid
    snake.grid.fill(GridEnum.EMPTY.value)

    # place head
    snake.head_x = head_x
    snake.head_y = head_y
    snake.grid[snake.head_x, snake.head_y] = GridEnum.HEAD.value

    # place food
    if not food:
        snake.grid[GRID_X-1, GRID_Y-1] = GridEnum.FOOD.value
    else:
        snake.grid[head_x + x_dir, head_y + y_dir] = GridEnum.FOOD.value

    x_old = snake.head_x
    y_old = snake.head_y
    snake.run_single(x_dir, y_dir)  # move right
    assert snake.head_x == x_old+x_dir
    assert snake.head_y == y_old+y_dir
    if food:
        assert snake.length == 0
    else:
        assert snake.length == 1


def test_spawn_food() -> None:
    """Test that food can be spawned at a random valid location."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    food_x = np.zeros((10,))
    food_y = np.zeros((10,))
    snake.grid.fill(GridEnum.EMPTY.value)
    snake.grid[0, :] = GridEnum.HEAD.value
    snake.grid[1, :] = GridEnum.HEAD.value

    for food in range(100):
        food_x[food], food_y[food] = snake.spawn_food()
    assert np.amax(snake.grid) == 0  # check no food
    assert np.std(food_x) > 0  # sanity check that food positions are not 0
    assert np.std(food_y) > 0


@pytest.mark.parametrize("direction", list(Direction))
def test_check_game_over_right(direction: Direction) -> None:
    """Move snake right until game over."""
    x_dir, y_dir = SnakeRL(neural_net=NeuralNetwork(
    ), x_grid_size=GRID_X, y_grid_size=GRID_Y).direction_to_tuple(direction)
    snake = SnakeDummy(GRID_X, GRID_Y)

    snake.head_x = GRID_X-1 if x_dir == 1 else 0
    snake.head_y = GRID_Y-1 if y_dir == 1 else 0
    snake.grid[snake.head_x, snake.head_y] = GridEnum.HEAD.value
    snake.run_single(x_dir, y_dir)
    assert snake.game_over


def test_check_game_over_tail() -> None:
    """Move snake into tail to end game."""
    snake = SnakeDummy(4, 4)
    assert snake.head_x == snake.head_y == 0
    snake.grid[0, 1] = -2  # place food at (0,1)
    snake.food_x, snake.food_y = 0, 1
    snake.run_single(0, 1)  # move up and eat
    snake.grid[1, 1] = -2  # place food at (1,1)
    snake.food_x, snake.food_y = 1, 1
    snake.run_single(1, 0)  # move right and eat
    snake.grid[1, 0] = -2  # place food at (1,0)
    snake.food_x, snake.food_y = 1, 0
    snake.run_single(0, -1)  # move down and eat
    snake.run_single(-1, 0)  # move left into old tail spot
    assert not snake.game_over
    snake.run_single(1, 0)  # move right into tail
    assert snake.game_over
