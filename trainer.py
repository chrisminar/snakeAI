from typing import Tuple

import numpy as np
from numpy import typing as npt

from helper import Globe as globe
from neural_net import NeuralNetwork
from timer import Timer


###################
## trainer class ##
###################
class Trainer:
    """description of class"""

    def __init__(self, nn: NeuralNetwork) -> None:
        self.nn = nn

    def train(self, generation: int,
              inputs: npt.NDArray[np.int32],
              heads: npt.NDArray[np.int32],
              move_predictions: npt.NDArray[np.int32]) -> None:
        with Timer() as t:
            # get all permutations
            statesP, movesP, headsP = Trainer.permute_inputs(
                inputs, move_predictions, heads)

            statesP = np.reshape(
                statesP, (statesP.shape[0], statesP.shape[1], statesP.shape[1], 1))  # reshape

            # train on them
            self.nn.train(statesP, headsP, movesP, generation)

    @staticmethod
    def permute_inputs(states: npt.NDArray[np.int32],
                       moves: npt.NDArray[np.int32],
                       heads: npt.NDArray[np.int32],) -> Tuple[npt.NDArray[np.int32],
                                                               npt.NDArray[np.int32],
                                                               npt.NDArray[np.int32], ]:
        flipAxis = len(states.shape)-1

        # rotate 90
        stateR90 = np.rot90(states, 1, (1, 2))
        movesR90 = Trainer.rotate_moves(moves, 1)
        headsR90 = Trainer.rotate_moves(heads, 1)

        # rotate 180
        stateR180 = np.rot90(states, 2, (1, 2))
        movesR180 = Trainer.rotate_moves(moves, 2)
        headsR180 = Trainer.rotate_moves(heads, 2)

        # rotate 270
        stateR270 = np.rot90(states, 3, (1, 2))
        movesR270 = Trainer.rotate_moves(moves, 3)
        headsR270 = Trainer.rotate_moves(heads, 3)

        # flip left - right
        stateLR = np.flip(states, axis=flipAxis-1)
        movesLR = Trainer.flip_move_left_right(moves)
        headsLR = Trainer.flip_move_left_right(heads)

        # rotate lr 90
        stateLRR90 = np.rot90(stateLR, 1, (1, 2))
        movesLRR90 = Trainer.rotate_moves(movesLR, 1)
        headsLRR90 = Trainer.rotate_moves(headsLR, 1)

        # rotate lr 180
        stateLRR180 = np.rot90(stateLR, 2, (1, 2))
        movesLRR180 = Trainer.rotate_moves(movesLR, 2)
        headsLRR180 = Trainer.rotate_moves(headsLR, 2)

        # rotate lr 270
        stateLRR270 = np.rot90(stateLR, 3, (1, 2))
        movesLRR270 = Trainer.rotate_moves(movesLR, 3)
        headsLRR270 = Trainer.rotate_moves(headsLR, 3)

        stateOut = np.vstack([states, stateR90, stateR180, stateR270,
                             stateLR, stateLRR90, stateLRR180, stateLRR270])
        movesOut = np.vstack([moves, movesR90, movesR180, movesR270,
                             movesLR, movesLRR90, movesLRR180, movesLRR270])
        headsOut = np.vstack([heads, headsR90, headsR180, headsR270,
                             headsLR, headsLRR90, headsLRR180, headsLRR270])

        return stateOut, movesOut, headsOut

    @staticmethod
    def flip_move_left_right(moves: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        movesLR = np.copy(moves)
        movesLR[:, 3], movesLR[:, 1] = moves[:, 1], moves[:, 3]
        return movesLR

    @staticmethod
    def flip_move_up_down(moves: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        movesUD = np.copy(moves)
        movesUD[:, 0], movesUD[:, 2] = moves[:, 2], moves[:, 0]
        return movesUD

    @staticmethod
    def rotate_moves(moves: npt.NDArray[np.int32], quads: int) -> npt.NDArray[np.int32]:
        return np.roll(moves, -quads, axis=1)
