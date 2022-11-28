"""Test functionality of snake base class."""

import numpy as np

from helper import GRID_X, GRID_Y
from snake.snake import Snake


class SnakeDummy(Snake):
    """Dummy child of Snake to test the functions of the base class."""


def test_init() -> None:
    """Test snake initialization."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    assert not snake.game_over
    assert snake.grid.shape == np.zeros((GRID_X, GRID_Y)).shape

# TODO figure out how to test functions in an abstract base class

# TODO make this a pytest with parametrized


def test_run_single_no_food() -> None:
    """Run single step, do not eat."""
    snake = SnakeDummy(GRID_X, GRID_Y)  # TODO change to constant
    # TODO fix initialized grid (head always in same spot)
    x_old = snake.head_x
    y_old = snake.head_y
    snake.run_single(1, 0)  # move right
    assert snake.head_x == x_old+1
    assert snake.head_y == y_old
    assert snake.length == 0
    x_old = snake.head_x
    y_old = snake.head_y
    snake.run_single(-1, 0)  # move left
    assert snake.head_x == x_old-1
    assert snake.head_y == y_old
    assert snake.length == 0
    x_old = snake.head_x
    y_old = snake.head_y
    snake.run_single(0, 1)  # move up
    assert snake.head_x == x_old
    assert snake.head_y == y_old+1
    assert snake.length == 0
    x_old = snake.head_x
    y_old = snake.head_y
    snake.run_single(0, -1)  # move down
    assert snake.head_x == x_old
    assert snake.head_y == y_old-1
    assert snake.length == 0


def test_run_single_with_food() -> None:
    """Run a step where the snake eats."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    x_old = snake.head_x
    y_old = snake.head_y
    len_old = snake.length
    score = snake.score
    snake.food_x = x_old+1
    snake.food_y = y_old
    snake.grid[snake.food_x][snake.food_y] = -2
    snake.run_single(1, 0)  # move right
    assert snake.grid[x_old, y_old] == len_old+1
    assert snake.score > score
    assert snake.length == len_old+1


def test_spawn_food() -> None:
    """Test that food can be spawned at a random valid location."""
    snake = SnakeDummy(GRID_X, GRID_Y)  # TODO make constant
    i = np.zeros((10,))
    j = np.zeros((10,))
    # TODO fill up grid with body more
    for food in range(100):
        i[food], j[food] = snake.spawn_food()
    assert np.amax(snake.grid) == 0  # check no food
    assert np.std(i) > 0  # todo what is this
    assert np.std(j) > 0

# TODO change move conditions to happen based off grid size


def test_check_game_over_right() -> None:
    """Move snake right until game over."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    # head at 0,0
    # TODO assert head at 0,0
    snake.run_single(1, 0)  # head at 1,0
    assert not snake.game_over
    snake.run_single(1, 0)  # head at 2,0
    assert not snake.game_over
    snake.run_single(1, 0)  # head at 3,0
    assert not snake.game_over
    snake.run_single(1, 0)  # head at 4,0 (dead)
    assert snake.game_over


def test_check_game_over_left() -> None:
    """Move snake left until game over."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    # head at 0,0
    # TODO assert heat at 0,0
    snake.run_single(-1, 0)  # head at -1,0
    assert snake.game_over

# TODO change name to reflect this moving the snake down


def test_check_game_over_up() -> None:
    """Move snake up until game over."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    # TODO assert head at 0,0
    # head at 0,0
    snake.run_single(0, 1)  # head at 0,1
    assert not snake.game_over
    snake.run_single(0, 1)  # head at 0,2
    assert not snake.game_over
    snake.run_single(0, 1)  # head at 0,3
    assert not snake.game_over
    snake.run_single(0, 1)  # head at 0,4 (dead)
    assert not snake.game_over

# TODO CHANGE NAME


def test_check_game_over_down() -> None:
    """Move snake down until game over."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    # head at 0,0
    # todo assert head at -,0
    assert not snake.game_over
    snake.run_single(0, -1)  # head at 0,-1
    assert not snake.game_over


def test_check_game_over_tail() -> None:
    """Move snake into tail to end game."""
    snake = SnakeDummy(GRID_X, GRID_Y)
    # todo assert head at 00
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
