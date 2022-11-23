from pathlib import Path

import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt

from neuralNet import NeuralNetwork as nn
from selfPlay import SelfPlay


def run_a_sample(checkpoint: Path) -> None:
    newnn = nn()
    newnn.load(checkpoint)
    spc = SelfPlay(newnn)
    states, heads, scores, ids, moves = spc.play_games(0, 0, 1)

    fig = plt.figure()
    ax = plt.axes(xlim=(-0.5, 3.5), ylim=(-0.5, 3.5))
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    ims = []
    for i in range(len(states)):
        im = plt.imshow(states[i], animated=True)
        title = ax.text(0.5, 1.05, "{}".format(i),
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, )
        ims.append([im, title])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=False)
    plt.show()


def gen_compare() -> None:

    states = []
    generation = [0, 100, 200, 383]
    for i in range(4):
        newnn = nn()
        newnn.load('saves/generation_{}.ckpt'.format(generation[i]))
        spc = SelfPlay(newnn)
        if i < 3:
            state, head, score, id, move = spc.play_games(0, 0, 1)
        else:
            state, head, score, id, move = spc.play_games(0, 0, 50)
        state = find_best(state, score, id)
        states.append(state)
        print('done with generation {}'.format(i))

    gamelength = []
    for i in range(4):
        gamelength.append(len(states[i]))

    for i in range(np.max(gamelength)):
        fig = plt.figure()
        for j in range(4):
            ax = plt.subplot(2, 2, 1+j)
            plt.xlim([-0.5, 3.5])
            plt.ylim([-0.5, 3.5])
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            if i < gamelength[j]:
                ax.imshow(states[j][i])
                plt.title('Generation {}, move {}'.format(generation[j], i))
            else:
                ax.imshow(states[j][-1])
                plt.title('Generation {}, move {}'.format(
                    generation[j], gamelength[j]))
        fig.savefig('compare/compare{}.png'.format(i))
        plt.close()


def find_best(states: npt.NDArray[np.int32],
              scores: npt.NDArray[np.int32],
              ids: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    # get indexes
    idxStart = np.argmax(scores)
    idx = idxStart
    index = []
    flag = True
    while flag:
        index.append(idx)
        idx += 1
        if idx >= len(scores):
            flag = False
        elif ids[idx] != ids[idxStart]:
            flag = False

    mask = np.zeros(len(ids), dtype=bool)
    mask[index] = True
    state = states[mask]
    return state


if __name__ == "__main__":
    # run_a_sample('saves/generation_383.ckpt')
    gen_compare()
