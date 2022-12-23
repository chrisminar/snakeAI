"""Holder class for outputs form playing snake games."""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy import typing as npt

from snake.big_snake import ParSnake as BigSnake
from training.helper import (GRID_X, GRID_Y, NUM_GAMES_PER_BATCH,
                             NUM_SELF_PLAY_GAMES, NUM_TRAINING_GAMES,
                             USE_EXPLORATION_CUTOFF, get_perf)
from training.neural_net import NeuralNetwork

LOGGER = logging.getLogger("terminal")


class PlayBig:
    """Play games in parallel."""

    def __init__(self, neural_network: NeuralNetwork) -> None:
        """Initialize game player.

        Args:
            neural_network (NeuralNetwork): Neural network to play with.
        """
        self.neural_net = neural_network

    def play_games(self,
                   *,
                   start_id: int = 0,
                   minimum_score: Optional[float] = None,
                   exploratory: bool = False,
                   num_games: int = NUM_GAMES_PER_BATCH,
                   best_generation_score: float = 0.0
                   ) -> Tuple[npt.NDArray[np.int32],
                              npt.NDArray[np.bool8],
                              npt.NDArray[np.int32],
                              npt.NDArray[np.int32],
                              npt.NDArray[np.float32],
                              float]:
        """Play some games.

        Args:
            start_id (int, optional): Uniquye id of first game to be played.
            neural_net (NeuralNetwork): Neural network to play games with.
            minimum_score (int, optional): Don't accept games below this score.
            exploratory (bool, optional): Should the snake preform exploratory moves?
            best_generation_score (float, optional): Best score.

        Returns:
            (npt.NDArray[np.uint32]): game states
            (npt.NDArray[npt.bool8]): heads
            (npt.NDArray[np.int32]): scores
            (npt.NDArray[np.int32]): game ids
            (npt.NDArray[np.float32]): predictions
            (float): performance
        """
        game_player = BigSnake(neural_net=self.neural_net,
                               exploratory=exploratory and (
                                   minimum_score is not None and minimum_score < USE_EXPLORATION_CUTOFF),
                               num_games=num_games)
        game_player.play(best_generation_score=best_generation_score)
        state, head, score, game_id, move = game_player.aggregate_results()

        pre_purge_score = get_perf(
            scores=score, ids=game_id, gen=0, plot=False)
        LOGGER.info("Mean score before purging is %02f", pre_purge_score)

        if minimum_score is not None:
            idx_above_minimum_score = score > minimum_score

            state = state[idx_above_minimum_score]
            head = head[idx_above_minimum_score]
            score = score[idx_above_minimum_score]
            game_id = game_id[idx_above_minimum_score] + start_id
            move = move[idx_above_minimum_score]

            LOGGER.debug(
                "Played %d games above minimum score(%02f) in %d attempts", np.unique(game_id).size, minimum_score, NUM_TRAINING_GAMES)

        return state, head, score, game_id, move.astype(np.float32), pre_purge_score
