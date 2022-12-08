"""Test big snake."""

from typing import Optional, Tuple

import numpy as np
import pytest
from numpy import typing as npt

from snake.big_snake import (ParSnake, choose_valids, directions_to_tuples,
                             get_random_valid)
from training.helper import GRID_X, GRID_Y, SCORE_PER_FOOD, Direction, GridEnum

XMAX = GRID_X-1
YMAX = GRID_Y-1

# pylint: disable=protected-access


def test_get_random_valid() -> None:
    """Choose a random valid position when the grid has valid positions."""
    array = np.arange(
        GRID_X*GRID_Y*2, dtype=np.int32).reshape((2, GRID_Y, GRID_X))
    bools = array % (GRID_X*GRID_Y) == GRID_X + \
        1  # one true value per at (1,1)
    for grid in range(2):
        indicies = get_random_valid(bools[grid])
        assert indicies.shape == (2, )
        assert np.all(indicies == 1)


def test_get_random_valid_with_backup() -> None:
    """Chose a random valid position when the grid has a valid position and there is a backup position."""
    array = np.full((GRID_Y, GRID_X), False, dtype=np.bool8)
    backup = array.copy()
    backup[1, 3] = True  # some random spot
    indicies = get_random_valid(array, backup)
    assert indicies.shape == (2, )
    np.array_equal(indicies, (1, 3))


@pytest.mark.parametrize("backup", [np.full((GRID_X, GRID_Y), False, dtype=np.bool8), None])
def test_get_random_valid_without_backup(backup: Optional[npt.NDArray[np.bool8]]) -> None:
    """Chose a random valid position when the grid has a valid position and there is no backup position."""
    array = np.full((GRID_Y, GRID_X), False, dtype=np.bool8)
    indicies = get_random_valid(array, backup)
    assert indicies.shape == (2, )
    np.array_equal(indicies, (0, 0))


def test_choose_valids() -> None:
    """Random valid locations chosen."""
    array = np.arange(
        GRID_X*GRID_Y*2, dtype=np.int32).reshape((2, GRID_Y, GRID_X))
    bools = array % (GRID_X*GRID_Y) == GRID_X + \
        1  # one true value per at (1,1)

    valids = choose_valids(bools)
    x_idx, y_idx = valids[:, 1], valids[:, 0]
    assert x_idx.shape == (2, )
    assert y_idx.shape == (2, )
    assert np.all(x_idx == 1)
    assert np.all(y_idx == 1)


@pytest.mark.parametrize("headx, heady, truth, foodx, foody", [
    (0, 0, (0, 1, 1, 0), XMAX, YMAX),  # down and right are free
    (0, 0, (0, 1, 1, 0), 0, 1),  # down free, right food
    (1, YMAX, (1, 1, 0, 1), 0, 0),  # down not free
    (XMAX, 1, (1, 0, 1, 1), 0, 0),  # right not free
    (0, YMAX, (1, 1, 0, 0), 1, 1),  # lower left corner
    (XMAX, 0, (0, 0, 1, 1), 1, 1),  # upper right corner
    (XMAX, YMAX, (1, 0, 0, 1), 1, 1),  # lower right corner
    (1, 0, (0, 1, 1, 1), 1, 1),  # top wall
    (XMAX, 1, (1, 0, 1, 1), 1, 1),  # right wall
    (1, YMAX, (1, 1, 0, 1), 1, 1),  # bot wall
    (0, 1, (1, 1, 1, 0), 1, 1),  # left wall
    (2, 2, (1, 1, 1, 1), 1, 1)  # middle
])
def test_convert_heads(headx: int, heady: int, truth: Tuple[int, ...], foodx: int, foody: int) -> None:
    """Convert head with no food and some walls."""
    snake = ParSnake(num_games=5)
    # make grid empty
    snake.grid.fill(-1)

    # head
    snake.heads_x.fill(headx)
    snake.heads_y.fill(heady)
    snake.grid[:, snake.heads_y, snake.heads_x] = GridEnum.HEAD.value

    # food
    snake.grid[:, foody, foodx] = GridEnum.FOOD.value

    is_frees = snake.convert_heads()
    for is_free in is_frees:
        np.testing.assert_equal(np.array(truth, dtype=np.bool8), is_free)


@pytest.mark.parametrize("direction", list(Direction))
def test_head_tracker(direction: Direction) -> None:
    """Move snake right until game over."""
    snake = ParSnake(num_games=5)

    xy_index = directions_to_tuples(np.full((5,), direction.value))

    head_x = GRID_X-1 if xy_index[0, 0] == 1 else 0
    head_y = GRID_Y-1 if xy_index[0, 1] == 1 else 0

    snake._reset(head_x=head_x, head_y=head_y)

    snake._snake_head_tracker_update(np.full((5,), direction.value))
    assert snake.lengths.size == 0  # should be 0 left afterkilling every game


@pytest.mark.parametrize("ate", [True, False])
def test_ate_this_turn(ate: bool) -> None:
    """Snake eating updates arrays correctly.

    Args:
        ate (bool): Did the snake eat?
    """
    snake = ParSnake(num_games=5)

    food_x = 2 if ate else 3
    food_y = 1
    snake._reset(head_x=1, head_y=1, food_x=food_x, food_y=food_y)
    snake.heads_x.fill(2)  # simulate moving one to the right
    snake.heads_x[-1] = 0  # simulate one head moving one to the left
    snake._snake_ate_this_turn()
    if ate:
        assert snake.lengths.size == 5
        assert np.all(snake.scores[0:-1] == SCORE_PER_FOOD)
        assert np.all(snake.lengths[0:-1] == 1)
        assert np.all(snake.moves_since_food[0:-1] == 0)
        assert snake.scores[-1] == 0
        assert snake.lengths[-1] == 0
        assert snake.moves_since_food[-1] == 1
    else:
        assert snake.lengths.size == 5
        assert np.all(snake.scores == 0)
        assert np.all(snake.lengths == 0)
        assert np.all(snake.moves_since_food == 1)


@pytest.mark.parametrize("ate", [True, False])
def test_update_grid(ate) -> None:
    """Grid updated correctly.

    Args:
        ate (_type_): Did the snake eat?
    """
    snake = ParSnake(num_games=5)
    food_x = 2 if ate else 3
    food_y = 1
    snake._reset(head_x=1, head_y=1, food_x=food_x, food_y=food_y)
    snake.heads_x.fill(2)  # simulate moving one to the right
    snake.heads_x[-1] = 0  # simulate one head moving one to the left

    snake._update_grid()

    # new location has head value
    if ate:
        assert np.all(snake.grid[snake.grid0[:-1], 1, 2] ==
                      GridEnum.FOOD.value)  # will not have updated yet
    else:
        assert np.all(snake.grid[snake.grid0[:-1], 1, 2] ==
                      GridEnum.EMPTY.value)  # will not have updated yet
    assert snake.grid[snake.grid0[-1], 1, 0] == GridEnum.EMPTY.value

    # old location is emtpy if it didn't eat
    if not ate:
        assert np.all(snake.grid[snake.grid0, 1, 1] == GridEnum.EMPTY.value)
    else:
        assert np.all(snake.grid[snake.grid0[:-1], 1, 1]
                      == GridEnum.HEAD.value+1)
        assert snake.grid[snake.grid0[-1], 1, 0] == GridEnum.EMPTY.value


def test_spawn_food() -> None:
    """Food spawned in valid locations."""
    snake = ParSnake(num_games=36)
    snake._reset(head_x=XMAX, head_y=YMAX, food_x=XMAX, food_y=YMAX-1)

    for x_idx in range(XMAX):
        for y_idx in range(YMAX):
            idx = y_idx*GRID_X+x_idx
            snake.heads_x[idx] = x_idx
            snake.heads_y[idx] = y_idx
            snake.food_xs[idx] = x_idx
            snake.food_ys[idx] = y_idx
            snake.grid[idx].fill(GridEnum.EMPTY.value)
            snake.grid[idx, y_idx, x_idx] = GridEnum.FOOD.value

    snake._spawn_food()

    for y_idx in range(GRID_Y):
        for x_idx in range(GRID_X):
            idx = y_idx*GRID_X+x_idx
            if x_idx == XMAX or y_idx == YMAX:
                # should only be one food
                assert np.sum(snake.grid[idx] == GridEnum.FOOD.value) == 1
                assert snake.grid[idx, snake.food_ys[idx],
                                  snake.food_xs[idx]] == GridEnum.FOOD.value
                assert snake.food_xs[idx] == XMAX
                assert snake.food_ys[idx] == YMAX-1
            else:
                # should be 2 foods
                assert np.sum(snake.grid[idx] == GridEnum.FOOD.value) == 2
                assert not ((snake.food_xs[idx] == x_idx)
                            and (snake.food_ys[idx] == y_idx))


def test_update_head() -> None:
    """Head postion updated after move."""
    snake = ParSnake(num_games=5)
    snake.grid.fill(GridEnum.EMPTY.value)
    head_xs = [0, 0, XMAX+1, -1, 0]
    head_ys = [YMAX+1, -1, 0, 0, 0]
    for i, (head_x, head_y) in enumerate(zip(head_xs, head_ys)):
        snake.heads_x[i] = head_x
        snake.heads_y[i] = head_y

    snake._update_head()

    assert snake.lengths.size == 1
    assert snake.grid[0, 0, 0] == GridEnum.HEAD.value
