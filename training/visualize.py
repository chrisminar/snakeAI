"""Visualize training."""
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt

from training.helper import GRID_X, GRID_Y
from training.neural_net import NeuralNetwork
from training.play_games import PlayBig as PlayGames

LOGGER = logging.getLogger("terminal")


def run_a_sample(checkpoint: Path) -> None:
    """Run one game from a train checkpoint.

    Args:
        checkpoint (Path): Checkpoint path.
    """
    neural_net = NeuralNetwork()
    neural_net.load(checkpoint)
    spc = PlayGames(neural_net)
    states, _, _, _, _ = spc.play_games(num_games=1)

    fig = plt.figure()
    axis = plt.axes(xlim=(-0.5, 3.5), ylim=(-0.5, 3.5))
    axis.axes.get_yaxis().set_visible(False)
    axis.axes.get_xaxis().set_visible(False)

    images = []
    for i, state in enumerate(states):
        image = plt.imshow(state, animated=True)
        title = axis.text(0.5, 1.05, str(i),
                          size=plt.rcParams["axes.titlesize"],
                          ha="center", transform=axis.transAxes, )
        images.append([image, title])

    animation.ArtistAnimation(fig, images, interval=200, blit=False)
    plt.show()


def gen_compare(generations: Tuple[int, int, int, int] = (0, 100, 200, 383), n=50) -> None:
    """Compare multiple generations."""
    states = []
    for i, generation in enumerate(generations):
        neural_net = NeuralNetwork()
        neural_net.load(Path(f'media/saves/generation_{generation}.ckpt'))
        spc = PlayGames(neural_net)
        state, _, score, ids, _ = spc.play_games(num_games=n)
        state = find_best(states=state, scores=score, ids=ids)
        states.append(state)
        LOGGER.info("Done with generation %d", generation)

    game_lengths = [len(state) for state in states]

    for i in range(np.max(game_lengths)):
        fig = plt.figure()
        for j, game_length in enumerate(game_lengths):
            axis = plt.subplot(2, 2, 1+j)
            plt.xlim([-0.5, GRID_X-0.5])
            plt.ylim([-0.5, GRID_Y-0.5])
            axis.axes.get_yaxis().set_visible(False)
            axis.axes.get_xaxis().set_visible(False)
            if i < game_length:
                axis.imshow(states[j][i])
                plt.title(f'Generation {generations[j]}, move {i}')
            else:
                axis.imshow(states[j][-1])
                plt.title(f'Generation {generations[j]}, move {game_length}')
        fig.savefig(f'media/compare/compare{i}.png')
        plt.close()


def find_best(*, states: npt.NDArray[np.int32],
              scores: npt.NDArray[np.int32],
              ids: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Find best game.

    Args:
        states (npt.NDArray[np.int32]): Game states.
        scores (npt.NDArray[np.int32]): Game scores.
        ids (npt.NDArray[np.int32]): Game ids.

    Returns:
        npt.NDArray[np.int32]: Best game state.
    """
    best_index = np.argmax(scores)
    best_id = ids[best_index]
    best_game_indicies = ids == best_id
    return states[best_game_indicies]


"""
from pathlib import Path

from training.visualize import gen_compare
gen_compare((40, 40, 40, 40))
"""
