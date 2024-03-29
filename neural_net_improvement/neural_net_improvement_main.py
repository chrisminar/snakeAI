"""Test and improve neural net architecture and hyperparameters."""
import logging
import re
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy import typing as npt

from training.helper import get_perf
from training.neural_net import NeuralNetwork
from training.play_games import PlayBig
from training.train_snake_reinforcement_learning import TrainRL
from training.trainer import train

LOGGER = logging.getLogger("terminal")


def load_training() -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.bool8], npt.NDArray[np.float32]]:
    """Load training data from a checkpoint.

    Raises:
        ValueError: If read shapes are bad.

    Returns:
        states (npt.NDArray[np.int32]): Pre-processed snake grids.
        heads (npt.NDArray[np.bool8]): Snake head availibliltiy.
        moves (npt.NDArray[np.float32]): The move that was chosen at each state.
    """
    saves = list(Path("./media/saves").glob("*.ckpt"))
    biggest_generation = max(
        [int(re.findall(r'\d+', save.name)[0]) for save in saves])
    biggest_path = Path(
        f"./media/saves/generation_{biggest_generation}.ckpt")

    LOGGER.info("Loading from path %s", biggest_path)

    trl = TrainRL()

    states = np.fromfile(biggest_path/"states.npy",
                         dtype=trl.game_states.dtype).reshape((-1, *trl.game_states.shape[1:]))
    heads = np.fromfile(biggest_path/"heads.npy",
                        dtype=trl.game_heads.dtype).reshape((-1, *trl.game_heads.shape[1:]))
    moves = np.fromfile(biggest_path/"moves.npy",
                        dtype=trl.moves.dtype).reshape((-1, *trl.moves.shape[1:]))

    if not states.shape[0] == heads.shape[0] == moves.shape[0]:
        LOGGER.info("State: %d, head: %d, moves: %d",
                    states.shape, heads.shape, moves.shape)
        raise ValueError("Failed to import training data with correct shapes.")
    return states, heads, moves


def play_once(neural_net: NeuralNetwork = NeuralNetwork()) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Play one set of games.

    Args:
        neural_net (NeuralNetwork, optional): Neural network to use.. Defaults to NeuralNetwork().

    Returns:
        Tuple[npt.NDArray[np.int32, np.int32]]: Game scores and ids
    """
    spc = PlayBig(neural_network=neural_net)

    _, __, scores, ids, ____, _____ = spc.play_games(
        start_id=0, minimum_score=-1000, num_games=10_000, exploratory=False)
    return scores, ids


def test_training(num_attempts: int = 2):
    """Test training of the nerual net.

    Args:
        num_attempts (int, optional): Number of attempts to train on. Defaults to 2.
    """
    states, heads, moves = load_training()

    for _ in range(num_attempts):
        neural_net = train(generation=0, states=states,
                           heads=heads, predictions=moves, verbose=1, permute_input=False)
        scores, ids = play_once(neural_net)
        mean = get_perf(scores, ids, 0)
        LOGGER.info("Mean score of %02f", mean)
