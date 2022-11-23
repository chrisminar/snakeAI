"""Train neural network on best games."""
from typing import Tuple

import numpy as np
from numpy import typing as npt

from helper import Timer
from neural_net import NeuralNetwork


class Trainer:
    """Train neural network on game data."""

    def __init__(self, nn: NeuralNetwork) -> None:
        self.nn = nn

    def train(self, generation: int,
              inputs: npt.NDArray[np.int32],
              heads: npt.NDArray[np.int32],
              move_predictions: npt.NDArray[np.int32]) -> None:
        with Timer() as t:
            # get all permutations
            permuted_states, permuted_moves, permuted_heads = Trainer.permute_inputs(
                inputs, move_predictions, heads)

            permuted_states = np.reshape(
                permuted_states, (permuted_states.shape[0], permuted_states.shape[1], permuted_states.shape[1], 1))  # reshape

            # train on them
            self.nn.train(permuted_states, permuted_heads,
                          permuted_moves, generation)

    @staticmethod
    def permute_inputs(states: npt.NDArray[np.int32],
                       moves: npt.NDArray[np.int32],
                       heads: npt.NDArray[np.int32],) -> Tuple[npt.NDArray[np.int32],
                                                               npt.NDArray[np.int32],
                                                               npt.NDArray[np.int32], ]:
        flip_axis = len(states.shape)-1

        # rotate 90
        state_r90 = np.rot90(states, 1, (1, 2))
        moves_r90 = Trainer.rotate_moves(moves, 1)
        heads_r90 = Trainer.rotate_moves(heads, 1)

        # rotate 180
        state_r180 = np.rot90(states, 2, (1, 2))
        moves_r180 = Trainer.rotate_moves(moves, 2)
        heads_r180 = Trainer.rotate_moves(heads, 2)

        # rotate 270
        state_r270 = np.rot90(states, 3, (1, 2))
        moves_r270 = Trainer.rotate_moves(moves, 3)
        heads_r270 = Trainer.rotate_moves(heads, 3)

        # flip left - right
        state_LR = np.flip(states, axis=flip_axis-1)
        moves_LR = Trainer.flip_move_left_right(moves)
        heads_LR = Trainer.flip_move_left_right(heads)

        # rotate lr 90
        state_LR_r90 = np.rot90(state_LR, 1, (1, 2))
        moves_LR_r90 = Trainer.rotate_moves(moves_LR, 1)
        heads_LR_r90 = Trainer.rotate_moves(heads_LR, 1)

        # rotate lr 180
        state_LR_r180 = np.rot90(state_LR, 2, (1, 2))
        moves_LR_r180 = Trainer.rotate_moves(moves_LR, 2)
        heads_LR_r180 = Trainer.rotate_moves(heads_LR, 2)

        # rotate lr 270
        state_LR_r270 = np.rot90(state_LR, 3, (1, 2))
        moves_LR_r270 = Trainer.rotate_moves(moves_LR, 3)
        heads_LR_r270 = Trainer.rotate_moves(heads_LR, 3)

        state_out = np.vstack([states, state_r90, state_r180, state_r270,
                               state_LR, state_LR_r90, state_LR_r180, state_LR_r270])
        moves_out = np.vstack([moves, moves_r90, moves_r180, moves_r270,
                               moves_LR, moves_LR_r90, moves_LR_r180, moves_LR_r270])
        heads_out = np.vstack([heads, heads_r90, heads_r180, heads_r270,
                               heads_LR, heads_LR_r90, heads_LR_r180, heads_LR_r270])

        return state_out, moves_out, heads_out

    @staticmethod
    def flip_move_left_right(moves: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        moves_LR = np.copy(moves)
        moves_LR[:, 3], moves_LR[:, 1] = moves[:, 1], moves[:, 3]
        return moves_LR

    @staticmethod
    def flip_move_up_down(moves: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        moves_UD = np.copy(moves)
        moves_UD[:, 0], moves_UD[:, 2] = moves[:, 2], moves[:, 0]
        return moves_UD

    @staticmethod
    def rotate_moves(moves: npt.NDArray[np.int32], quads: int) -> npt.NDArray[np.int32]:
        return np.roll(moves, -quads, axis=1)
