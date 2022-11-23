"""Holder class for outputs form playing snake games."""

from typing import Callable

import numpy as np
from numpy import typing as npt

from neural_net import NeuralNetwork

#####################
## self play class ##
#####################


class PlayGames:
    """description of class"""

    def __init__(self, neural_network: NeuralNetwork) -> None:
        self.game_states = []
        self.prediction = []
        self.game_id = []
        self.scores = []
        self.heads = []
        self.neural_net = neural_network
        self.gamestate_to_nn: Callable[[npt.NDArray[np.int32]],
                                       npt.NDArray[np.int32]] = np.vectorize(grid_val_to_neural_net)


def grid_val_to_neural_net(input: int) -> int:
    """Convert input snake grid value to nn value."""
    if input == -1:  # empty -1 -> 0
        return 0
    elif input == -2:  # food -2 -> -1
        return -1
    else:  # head 0 -> 1, body positive -> 1
        return 1
