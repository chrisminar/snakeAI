"""Test neural network."""

import pytest

from snake.snake import Direction
from snake.snake_reinforcement_learning import SnakeRL
from training.helper import GRID_X, GRID_Y
from training.neural_net import NeuralNetwork
from training.play_games import PlayGames


def test_init() -> None:
    """Initialize neural net and check output layer shape."""
    neural_net = NeuralNetwork()
    assert neural_net.model.get_layer('policy').output_shape, (None, 4)


@pytest.mark.skip()
def test_display_model() -> None:
    """Test model display."""
    neural_network = NeuralNetwork()
    neural_network.disp_model()


@pytest.mark.parametrize("head_x, head_y",
                         [(0, 0),  # upper left
                          (GRID_X-1, 0),  # upper right
                             (GRID_X-1, GRID_Y-1),  # lower right BAD
                             (0, GRID_Y-1),  # lower left
                             (1, 0),  # top
                             (GRID_X-1, 1),  # right BAD
                             (1, GRID_Y-1),  # bot BAD
                             (0, 1)])  # left
@pytest.mark.parametrize("food", [True, False])
def test_evaluate(head_x: int, head_y: int, food: bool) -> None:
    """Evalute should only output non-lethal move options, if they exist.

    Args:
        head_x (int): Head position.
        head_y (int): Head position.
        food (bool): Is there food near the head?

    Raises:
        NotImplementedError: _description_
    """
    neural_net = NeuralNetwork()

    snake = SnakeRL(neural_net=neural_net,
                    x_grid_size=GRID_X, y_grid_size=GRID_Y)

    food_x = head_x+1 if head_x == 0 else head_x-1
    if food:
        food_y = head_y
    else:
        food_y = head_y+1 if head_y == 0 else head_y-1

    snake._reset(head_x=head_x, head_y=head_y, food_x=food_x, food_y=food_y)

    pre_processed_grid = PlayGames(neural_net).gamestate_to_nn(snake.grid)
    head_view = snake.convert_head()

    assert head_view[Direction.UP.value] if head_y > 0 else not head_view[Direction.UP.value]
    assert head_view[Direction.RIGHT.value] if head_x < GRID_X - \
        1 else not head_view[Direction.RIGHT.value]
    assert head_view[Direction.DOWN.value] if head_y < GRID_Y - \
        1 else not head_view[Direction.DOWN.value]
    assert head_view[Direction.LEFT.value] if head_x > 0 else not head_view[Direction.LEFT.value]

    policy = neural_net.evaluate(state=pre_processed_grid, head=head_view)[0]

    assert policy[Direction.UP.value] == 0 if head_y == 0 else policy[Direction.UP.value] != 0
    assert policy[Direction.RIGHT.value] == 0 if head_x == GRID_X - \
        1 else policy[Direction.RIGHT.value] != 0
    assert policy[Direction.DOWN.value] == 0 if head_y == GRID_Y - \
        1 else policy[Direction.DOWN.value] != 0
    assert policy[Direction.LEFT.value] == 0 if head_x == 0 else policy[Direction.LEFT.value] != 0


@pytest.mark.skip()
def test_load():
    """To do."""
    raise NotImplementedError
