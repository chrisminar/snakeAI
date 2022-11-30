"""Test functionality of rl snake class."""

from typing import Tuple

import numpy as np
import pytest

from snake.snake import Direction, GridEnum
from snake.snake_reinforcement_learning import SnakeRL
from training.helper import GRID_X, GRID_Y
from training.neural_net import NeuralNetwork
from training.play_games import PlayGames


# pylint: disable=protected-access
def test_init() -> None:
    """Test snake rl init."""
    snake = SnakeRL(neural_net=NeuralNetwork(),
                    x_grid_size=GRID_X, y_grid_size=GRID_Y)
    assert hasattr(snake, 'neural_net')


def test_evaluate_next() -> None:
    """Test nn evaluation."""
    neural_net = NeuralNetwork()
    snake = SnakeRL(neural_net=neural_net,
                    x_grid_size=GRID_X, y_grid_size=GRID_Y)
    snake._reset()  # place head at 0,0
    direction, move_array, head = snake.evaluate_next_step(
        PlayGames(neural_net).gamestate_to_nn)
    assert 0 <= direction < 4, 'invalid direction output'
    assert np.argmax(move_array) <= 3, 'invalid direction output'
    assert np.argmax(
        move_array) == direction, 'direction doesn\'t match move array'
    assert head[Direction.UP.value] == 0, 'up is free'
    assert head[Direction.RIGHT.value] == 1, 'right is free'
    assert head[Direction.DOWN.value] == 1, 'down is free'
    assert head[Direction.LEFT.value] == 0, 'left is free'


def test_play() -> None:
    """Play games."""
    neural_net = NeuralNetwork()
    snake = SnakeRL(neural_net=neural_net,
                    x_grid_size=GRID_X, y_grid_size=GRID_Y)
    games = PlayGames(neural_net)
    snake.play(games.gamestate_to_nn)
    assert snake.game_over
    assert len(snake.move_list) > 0


@pytest.mark.parametrize("headx, heady, truth, foodx, foody", [
    (0, 0, (0, 1, 1, 0), 3, 3),  # down and right are free
    (0, 0, (0, 1, 1, 0), 0, 1),  # down free, right food
    (1, 3, (1, 1, 0, 1), 0, 0),  # down not free
    (3, 1, (1, 0, 1, 1), 0, 0)])  # right not free
def test_convert_head(headx: int, heady: int, truth: Tuple[int, ...], foodx: int, foody: int) -> None:
    """Convert head with no food and some walls."""
    neural_net = NeuralNetwork()
    snake = SnakeRL(neural_net=neural_net,
                    x_grid_size=4, y_grid_size=4)
    # make grid empty
    snake.grid.fill(-1)

    # head
    snake.head_x = headx
    snake.head_y = heady
    snake.grid[snake.head_x, snake.head_y] = GridEnum.HEAD.value

    # food
    snake.grid[foodx, foody] = GridEnum.FOOD.value

    is_free = snake.convert_head()
    np.testing.assert_equal(np.array(truth, dtype=np.bool8), is_free)
