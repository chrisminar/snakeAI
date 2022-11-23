"""Holder class for outputs form playing snake games."""

from typing import Callable, List, Tuple

import numpy as np
from numpy import typing as npt

from helper import GRID_X, GRID_Y, NUM_SELF_PLAY_GAMES, Timer
from neural_net import NeuralNetwork
from snake.snake_reinforcement_learning import SnakeRL as snake


class PlayGames:
    """Play many games with a trained neural network."""

    def __init__(self, neural_network: NeuralNetwork) -> None:
        """Initialize game player.

        Args:
            neural_network (NeuralNetwork): Neural network to play with.
        """
        self.game_states: List[npt.NDArray[np.int32]] = []
        self.prediction: List[npt.NDArray[np.float32]] = []
        self.game_id: List[npt.NDArray[np.int32]] = []
        self.scores: List[npt.NDArray[np.int32]] = []
        self.heads: List[npt.NDArray[np.bool8]] = []
        self.neural_net = neural_network
        self.gamestate_to_nn: Callable[[npt.NDArray[np.int32]],
                                       npt.NDArray[np.int32]] = np.vectorize(grid_val_to_neural_net)

    def play_games(self,
                   start_id: int,
                   num_games: int = NUM_SELF_PLAY_GAMES) -> Tuple[npt.NDArray[np.uint32],
                                                                  npt.NDArray[npt.bool8],
                                                                  npt.NDArray[np.int32],
                                                                  npt.NDArray[np.int32],
                                                                  npt.NDArray[np.float32]]:
        """Play some games.

        Args:
            start_id (int): Uniquye id of first game to be played.
            num_games (int, optional): Number of games to play. Defaults to NUM_SELF_PLAY_GAMES.

        Returns:
            (npt.NDArray[np.uint32]): game states
            (npt.NDArray[npt.bool8]): heads
            (npt.NDArray[np.int32]): scores
            (npt.NDArray[np.int32]): game ids
            (npt.NDArray[np.float32]): predictions
        """
        for i in range(num_games):
            with Timer():
                g = snake(nn=self.neural_net,
                          sizeX=GRID_X, sizeY=GRID_Y)
                g.play(self.gamestate_to_nn)
                if len(g.moveList) > 1:
                    self.game_states.append(np.stack(g.stateList[:-1]))
                    self.heads.append(np.stack(g.headList[:-1]))
                    self.game_id.append(
                        np.full((len(g.stateList[:-1]), ), start_id+i))
                    self.prediction.append(np.array(g.moveList[:-1]))
                    self.scores.append(
                        np.full((len(g.stateList[:-1]), ), g.score))

        return self.gamestate_to_nn(np.concatenate(self.game_states)), np.concatenate(self.heads), np.concatenate(self.scores), np.concatenate(self.game_id), np.concatenate(self.prediction)


def grid_val_to_neural_net(grid_val: int) -> int:
    """Convert input snake grid value to nn value.

    Args:
        grid_val (int): Value from grid cell.

    Returns:
        int: Pre-processed grid cell value.
    """
    if grid_val == -1:  # empty -1 -> 0
        return 0
    if grid_val == -2:  # food -2 -> -1
        return -1
    # head 0 -> 1, body positive -> 1
    return 1
