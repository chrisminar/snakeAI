"""Test big snake."""

from typing import Optional

import numpy as np
import pytest
from numpy import typing as npt

from snake.big_snake import ParSnake
from training.helper import GRID_X, GRID_Y


def test_get_random_valid() -> None:
    """Choose a random valid position when the grid has valid positions."""
    snake = ParSnake(num_games=2)
    array = np.arange(
        GRID_X*GRID_Y*2, dtype=np.int32).reshape((2, GRID_Y, GRID_X))
    bools = array % (GRID_X*GRID_Y) == GRID_X + \
        1  # one true value per at (1,1)
    for grid in range(2):
        indicies = snake.get_random_valid(bools[grid])
        assert indicies.shape == (2, )
        assert np.all(indicies == 1)


def test_get_random_valid_with_backup() -> None:
    """Chose a random valid position when the grid has a valid position and there is a backup position."""
    snake = ParSnake(num_games=1)
    array = np.full((GRID_Y, GRID_X), False, dtype=np.bool8)
    backup = array.copy()
    backup[1, 3] = True  # some random spot
    indicies = snake.get_random_valid(array, backup)
    assert indicies.shape == (2, )
    np.array_equal(indicies, (1, 3))


@pytest.mark.parametrize("backup", [np.full((GRID_X, GRID_Y), False, dtype=np.bool8), None])
def test_get_random_valid_without_backup(backup: Optional[npt.NDArray[np.bool8]]) -> None:
    """Chose a random valid position when the grid has a valid position and there is no backup position."""
    snake = ParSnake(num_games=1)
    array = np.full((GRID_Y, GRID_X), False, dtype=np.bool8)
    indicies = snake.get_random_valid(array, backup)
    assert indicies.shape == (2, )
    np.array_equal(indicies, (0, 0))


def test_choose_valids() -> None:
    snake = ParSnake(num_games=1)
    array = np.arange(
        GRID_X*GRID_Y*2, dtype=np.int32).reshape((2, GRID_Y, GRID_X))
    bools = array % (GRID_X*GRID_Y) == GRID_X + \
        1  # one true value per at (1,1)

    valids = snake.choose_valids(bools)
    x, y = valids[:, 1], valids[:, 0]
    assert x.shape == (2, )
    assert y.shape == (2, )
    assert np.all(x == 1)
    assert np.all(y == 1)
