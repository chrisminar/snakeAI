"""Test functionality of rl snake class."""

from typing import Tuple

import numpy as np
import pytest

from snake.snake import GridEnum
from snake.snake_reinforcement_learning import SnakeRL
from training.helper import GRID_X, GRID_Y
from training.neural_net import NeuralNetwork
from training.play_games import PlayGames


def test_init() -> None:
    """Test snake rl init."""
    snake = SnakeRL(neural_net=NeuralNetwork(),
                    x_grid_size=GRID_X, y_grid_size=GRID_Y)
    assert hasattr(snake, 'nn')


def test_evaluate_next() -> None:
    """Test nn evaluation."""
    neural_net = NeuralNetwork()
    snake = SnakeRL(neural_net=neural_net,
                    x_grid_size=GRID_X, y_grid_size=GRID_Y)
    games = PlayGames(neural_net)
    direction, move_array, head = snake.evaluate_next_step(
        games.gamestate_to_nn)
    assert np.argmax(direction) >= 0, 'invalid direction output'
    assert np.argmax(move_array) <= 3, 'invalid direction output'
    assert np.argmax(
        move_array) == direction, 'direction doesn\'t match move array'
    assert head[0] == 1, 'up is free'
    assert head[1] == 1, 'right is free'
    assert head[2] == 0, 'down is free'
    assert head[3] == 0, 'left is free'


def test_play() -> None:
    """Play games."""
    neural_net = NeuralNetwork()
    snake = SnakeRL(neural_net=neural_net,
                    x_grid_size=GRID_X, y_grid_size=GRID_Y)
    games = PlayGames(neural_net)
    snake.play(games.gamestate_to_nn)
    assert not snake.game_over
    assert len(snake.move_list) > 0


@pytest.mark.parametrize("headx, heady, truth, foodx, foody", [
    (0, 0, (1, 1, 0, 0), 3, 3),  # up and right are free
    (0, 0, (1, 1, 0, 0), 0, 1),  # up free, right food
    (1, 3, (0, 1, 1, 1), 0, 0),  # up not free
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
