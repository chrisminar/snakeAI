"""Visualize training."""
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt

from training.neural_net import NeuralNetwork
from training.play_games import PlayGames

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


def gen_compare(generations: Tuple[int, int, int, int] = (0, 100, 200, 383)) -> None:
    """Compare multiple generations."""
    states = []
    for i, generation in enumerate(generations):
        neural_net = NeuralNetwork()
        neural_net.load(Path(f'media/saves/generation_{generation}.ckpt'))
        spc = PlayGames(neural_net)
        if i < 3:
            state, _, score, ids, _ = spc.play_games(num_games=1)
        else:
            state, _, score, ids, _ = spc.play_games(num_games=50)
        state = find_best(states=state, scores=score, ids=ids)
        states.append(state)
        LOGGER.info("Done with generation %d", generation)

    game_lengths = [len(state) for state in states]

    for i in range(np.max(game_lengths)):
        fig = plt.figure()
        for j, game_length in enumerate(game_lengths):
            axis = plt.subplot(2, 2, 1+j)
            plt.xlim([-0.5, 3.5])
            plt.ylim([-0.5, 3.5])
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
    # get indexes
    idx_start = np.argmax(scores)
    idx = idx_start
    index = []
    flag = True
    while flag:
        index.append(idx)
        idx += 1
        if idx >= len(scores):
            flag = False
        elif ids[idx] != ids[idx_start]:
            flag = False

    mask = np.zeros(len(ids), dtype=np.bool8)
    mask[index] = True
    state = states[mask]
    return state


"""
from pathlib import Path

from training.visualize import gen_compare, run_a_sample
run_a_sample(Path('media/saves/generation_75.ckpt'))
gen_compare((0, 25, 50, 75))
"""
