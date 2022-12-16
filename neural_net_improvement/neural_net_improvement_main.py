"""Test and improve neural net architecture."""
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

    states = np.fromfile(biggest_path/"states.bin",
                         dtype=trl.game_states.dtype).reshape((-1, *trl.game_states.shape[1:]))
    heads = np.fromfile(biggest_path/"heads.bin",
                        dtype=trl.game_heads.dtype).reshape((-1, *trl.game_heads.shape[1:]))
    moves = np.fromfile(biggest_path/"moves.bin",
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

    _, __, scores, ids, ____ = spc.play_games(
        start_id=0, minimum_score=-1000, exploratory=False)
    return scores, ids


def training_variance():
    """Understand how re-training effects NN performance."""
    states, heads, moves = load_training()
    for i in range(5):
        neural_net = train(generation=0, game_states=states,
                           heads=heads, move_predictions=moves, verbose=2)
        scores, ids = play_once(neural_net)
        mean = get_perf(scores, ids, i)
        LOGGER.info("Mean score of %02f", mean)


def test_a_nn():
    states, heads, moves = load_training()
    neural_net = NeuralNetwork()
    neural_net.train(states=states, heads=heads,
                     predictions=moves, generation=0, verbose=2)
    #neural_net = train(generation=0, game_states=states, heads=heads, move_predictions=moves, verbose=2)
    scores, ids = play_once(neural_net)
    mean = get_perf(scores, ids, 0)
    LOGGER.info("Mean score of %02f", mean)
# 2
# load everything
# load a new NN config
# run a training batch
# check mean score
