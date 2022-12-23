"""Test playing games."""

import numpy as np

from training.helper import GridEnum
from training.neural_net import NeuralNetwork
from training.play_games import PlayBig


def test_play_games() -> None:
    """Plame some games."""
    spc = PlayBig(NeuralNetwork())  # make self play class
    state, _, score, ids, prediction, _ = spc.play_games(num_games=2)

    print(state)
    print(score)
    print(ids)
    print(prediction)
    # test gamestate list
    assert state.shape[0] > 0

    # test gamescore
    assert score.shape[0] > 0

    # test gameid
    assert ids.shape[0] > 0
    assert np.max(ids) == 1

    # test prediction
    assert prediction.shape[0] > 0
    assert np.max(prediction) <= 3
    assert np.min(prediction) >= 0


def test_grid_2_neural_network() -> None:
    """Convert grid to neural network input."""
    neural_net = NeuralNetwork()
    grid = np.zeros((4, 4), dtype=np.int32) - 1  # set all values to empty
    for i in range(3):
        for j in range(3):
            # body/head (this won't actually make a valid snake, but that is ok for this test)
            grid[i, j] = i*3+j
    grid[3, 3] = -2  # make one value food

    processed_grid = neural_net.pre_process_input(grid.reshape(1, *grid.shape))

    head_location = grid == GridEnum.HEAD.value
    assert np.all(processed_grid[0, head_location, 0] == 1)
    assert np.all(processed_grid[0, np.logical_not(head_location), 0] == 0)

    food_location = grid == GridEnum.FOOD.value
    assert np.all(processed_grid[0, food_location, 1] == 1)
    assert np.all(processed_grid[0, np.logical_not(food_location), 1] == 0)

    empty_location = grid == GridEnum.EMPTY.value
    assert np.all(processed_grid[0, empty_location, 2] == 1)
    assert np.all(processed_grid[0, np.logical_not(empty_location), 2] == 0)

    body_location = grid >= GridEnum.BODY.value
    assert np.all(processed_grid[0, body_location, 3] == 1)
    assert np.all(processed_grid[0, np.logical_not(body_location), 3] == 0)
