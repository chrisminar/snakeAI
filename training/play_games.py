"""Holder class for outputs form playing snake games."""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy import typing as npt

from snake.snake import GridEnum
from snake.snake_reinforcement_learning import SnakeRL as snake
from training.helper import (GRID_X, GRID_Y, NUM_SELF_PLAY_GAMES,
                             NUM_TRAINING_GAMES, PreProcessedGrid)
from training.neural_net import NeuralNetwork

LOGGER = logging.getLogger("terminal")


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
                   *,
                   start_id: int = 0,
                   num_games: int = NUM_SELF_PLAY_GAMES,
                   minimum_score: Optional[float] = None,
                   exploratory: bool = False) -> Tuple[npt.NDArray[np.int32],
                                                       npt.NDArray[np.bool8],
                                                       npt.NDArray[np.int32],
                                                       npt.NDArray[np.int32],
                                                       npt.NDArray[np.float32]]:
        """Play some games.

        Args:
            start_id (int, optional): Uniquye id of first game to be played.
            num_games (int, optional): Number of games to play. Defaults to NUM_SELF_PLAY_GAMES.
            minimum_Score (int, optional): Don't accept games below this score.
            exploratory (bool, optional): Should the snake preform exploratory moves?

        Returns:
            (npt.NDArray[np.uint32]): game states
            (npt.NDArray[npt.bool8]): heads
            (npt.NDArray[np.int32]): scores
            (npt.NDArray[np.int32]): game ids
            (npt.NDArray[np.float32]): predictions
        """
        num_accepted = 0
        num_played = 0
        while num_accepted < num_games and not num_played > NUM_TRAINING_GAMES:
            games = snake(neural_net=self.neural_net,
                          exploratory=exploratory,
                          x_grid_size=GRID_X, y_grid_size=GRID_Y)
            games.play(self.gamestate_to_nn)
            num_played += 1
            if minimum_score is None or games.score > minimum_score:
                if len(games.move_list) == 1:
                    continue
                num_accepted += 1
                self.game_states.append(np.stack(games.state_list[:-1]))
                self.heads.append(np.stack(games.head_list[:-1]))
                self.game_id.append(
                    np.full((len(games.state_list[:-1]), ), start_id+num_accepted))
                self.prediction.append(np.array(games.move_list[:-1]))
                self.scores.append(
                    np.full((len(games.state_list[:-1]), ), games.score))
        LOGGER.debug(
            "Played %d games above minimum score in %d attempts", num_accepted, num_played)

        return (self.gamestate_to_nn(np.concatenate(self.game_states)),
                np.concatenate(self.heads),
                np.concatenate(self.scores),
                np.concatenate(self.game_id),
                np.concatenate(self.prediction))


def grid_val_to_neural_net(grid_val: int) -> int:
    """Convert input snake grid value to nn value.

    Args:
        grid_val (int): Value from grid cell.

    Returns:
        int: Pre-processed grid cell value.
    """
    if grid_val == GridEnum.FOOD.value:
        return PreProcessedGrid.FOOD.value
    if grid_val == GridEnum.EMPTY.value:
        return PreProcessedGrid.EMPTY.value
    return PreProcessedGrid.SNAKE.value
