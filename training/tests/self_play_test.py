"""Test playing games."""

import numpy as np

from training.helper import GridEnum, PreProcessedGrid
from training.neural_net import NeuralNetwork
from training.play_games import PlayGames


def test_play_games() -> None:
    """Plame some games."""
    spc = PlayGames(NeuralNetwork())  # make self play class
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
    spc = PlayGames(neural_net)  # make self play class
    grid = np.zeros((4, 4), dtype=np.int32) - 1  # set all values to empty
    for i in range(3):
        for j in range(3):
            # body/head (this won't actually make a valid snake, but that is ok for this test)
            grid[i, j] = i*3+j
    grid[3, 3] = -2  # make one value food

    processed_grid = spc.gamestate_to_nn(grid)

    assert np.all(
        processed_grid[grid >= GridEnum.HEAD.value] == PreProcessedGrid.SNAKE.value)
    assert np.all(
        processed_grid[grid == GridEnum.EMPTY.value] == PreProcessedGrid.EMPTY.value)
    assert np.all(
        processed_grid[grid == GridEnum.FOOD.value] == PreProcessedGrid.FOOD.value)
