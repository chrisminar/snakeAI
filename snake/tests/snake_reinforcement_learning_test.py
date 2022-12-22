"""Test functionality of rl snake class."""

from typing import Tuple

import numpy as np
import pytest

from snake.snake_reinforcement_learning import SnakeRL
from training.helper import GRID_X, GRID_Y, Direction, GridEnum
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
    (3, 1, (1, 0, 1, 1), 0, 0),  # right not free
    (0, 3, (1, 1, 0, 0), 1, 1),  # lower left corner
    (3, 0, (0, 0, 1, 1), 1, 1),  # upper right corner
    (3, 3, (1, 0, 0, 1), 1, 1),  # lower right corner
    (1, 0, (0, 1, 1, 1), 1, 1),  # top wall
    (3, 1, (1, 0, 1, 1), 1, 1),  # right wall
    (1, 3, (1, 1, 0, 1), 1, 1),  # bot wall
    (0, 1, (1, 1, 1, 0), 1, 1),  # left wall
    (2, 2, (1, 1, 1, 1), 1, 1)  # middle
])
def test_convert_head(headx: int, heady: int, truth: Tuple[int, ...], foodx: int, foody: int) -> None:
    """Convert head with no food and some walls."""
    neural_net = NeuralNetwork()
    snake = SnakeRL(neural_net=neural_net,
                    x_grid_size=4, y_grid_size=4, exploratory=False)
    # make grid empty
    snake.grid.fill(-1)

    # head
    snake.head_x = headx
    snake.head_y = heady
    snake.grid[snake.head_y, snake.head_x] = GridEnum.HEAD.value

    # food
    snake.grid[foody, foodx] = GridEnum.FOOD.value

    is_free = snake.convert_head()
    np.testing.assert_equal(np.array(truth, dtype=np.bool8), is_free)


@pytest.mark.parametrize("_", range(1000))
def test_victory(_: int) -> None:
    """Ensure game ends properly when neural net wins.

    Args:
        _ (int): randomness of NN initialization can effect output so try many times
    """
    neural_net = NeuralNetwork(x_size=4, y_size=4)
    snake = SnakeRL(neural_net=neural_net,
                    x_grid_size=4, y_grid_size=4, exploratory=False)

    snake._reset(head_x=0, head_y=0, food_x=0, food_y=1)
    # 0, 1, 2, 3
    # -2,6, 5, 4
    # -1 7, 8, 9
    # -1 12,11,10
    snake.grid = np.array([[0, 1, 2, 3],
                          [-2, 6, 5, 4],
                           [-1, 7, 8, 9],
                           [-1, 12, 11, 10]], dtype=np.int32)
    snake.length = 12
    new_direction, __, ___ = snake.evaluate_next_step()
    x_dir, y_dir = snake.direction_to_tuple(new_direction)
    assert x_dir == 0
    assert y_dir == 1
    snake.run_single(x_dir, y_dir)
    assert not snake.game_over

    snake.food_x = 0
    snake.food_y = 2
    snake.grid[snake.food_y, snake.food_x] = GridEnum.FOOD.value
    snake.grid[3, 0] = GridEnum.EMPTY.value
    new_direction, __, ___ = snake.evaluate_next_step()
    x_dir, y_dir = snake.direction_to_tuple(new_direction)
    assert x_dir == 0
    assert y_dir == 1
    snake.run_single(x_dir, y_dir)
    assert not snake.game_over

    new_direction, __, ___ = snake.evaluate_next_step()
    x_dir, y_dir = snake.direction_to_tuple(new_direction)
    assert x_dir == 0
    assert y_dir == 1
    snake.run_single(x_dir, y_dir)
    assert snake.game_over
