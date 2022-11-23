"""Train neural network on best games."""
from typing import Tuple

import numpy as np
from numpy import typing as npt

from helper import Timer
from neural_net import NeuralNetwork


def train(generation: int,
          game_states: npt.NDArray[np.int32],
          heads: npt.NDArray[np.bool8],
          move_predictions: npt.NDArray[np.float32]) -> None:
    """Permute input data and train on it.

    Args:
        generation (int): Generation number
        game_states (npt.NDArray[np.int32]): grids
        heads (npt.NDArray[np.int32]): head availiblity
        move_predictions (npt.NDArray[np.int32]): predicted moves
    """
    neural_net = NeuralNetwork()
    with Timer():
        # get all permutations
        permuted_states, permuted_heads, permuted_predictions = permute_inputs(
            game_states, move_predictions, heads)

        permuted_states = np.reshape(
            permuted_states, (permuted_states.shape[0], permuted_states.shape[1], permuted_states.shape[1], 1))  # reshape

        # train on permutations
        neural_net.train(permuted_states, permuted_heads,
                         permuted_predictions, generation)


def permute_inputs(states: npt.NDArray[np.int32],
                   heads: npt.NDArray[np.bool8],
                   predictions: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.int32],
                                                                  npt.NDArray[np.bool8],
                                                                  npt.NDArray[np.float32]]:
    """Permute inputs by flipping and rotating.

    Args:
        states (npt.NDArray[np.int32]): grids
        heads (npt.NDArray[np.bool8]): head moves
        predictions (npt.NDArray[np.float32]): move predictions

    Returns:
        Tuple[npt.NDArray[np.int32], npt.NDArray[np.bool8], npt.NDArray[np.float32]]: permuted inputs
    """
    flip_axis = len(states.shape)-1

    # rotate 90
    state_r90 = np.rot90(states, 1, (1, 2))
    moves_r90 = rotate_predictions(heads, 1)
    heads_r90 = rotate_predictions(predictions, 1)

    # rotate 180
    state_r180 = np.rot90(states, 2, (1, 2))
    moves_r180 = rotate_predictions(heads, 2)
    heads_r180 = rotate_predictions(predictions, 2)

    # rotate 270
    state_r270 = np.rot90(states, 3, (1, 2))
    moves_r270 = rotate_predictions(heads, 3)
    heads_r270 = rotate_predictions(predictions, 3)

    # flip left - right
    state_LR = np.flip(states, axis=flip_axis-1)
    moves_LR = flip_predictions_left_right(heads)
    heads_LR = flip_predictions_left_right(predictions)

    # rotate lr 90
    state_LR_r90 = np.rot90(state_LR, 1, (1, 2))
    moves_LR_r90 = rotate_predictions(moves_LR, 1)
    heads_LR_r90 = rotate_predictions(heads_LR, 1)

    # rotate lr 180
    state_LR_r180 = np.rot90(state_LR, 2, (1, 2))
    moves_LR_r180 = rotate_predictions(moves_LR, 2)
    heads_LR_r180 = rotate_predictions(heads_LR, 2)

    # rotate lr 270
    state_LR_r270 = np.rot90(state_LR, 3, (1, 2))
    moves_LR_r270 = rotate_predictions(moves_LR, 3)
    heads_LR_r270 = rotate_predictions(heads_LR, 3)

    state_out = np.vstack([states, state_r90, state_r180, state_r270,
                           state_LR, state_LR_r90, state_LR_r180, state_LR_r270])
    moves_out = np.vstack([heads, moves_r90, moves_r180, moves_r270,
                           moves_LR, moves_LR_r90, moves_LR_r180, moves_LR_r270])
    heads_out = np.vstack([predictions, heads_r90, heads_r180, heads_r270,
                           heads_LR, heads_LR_r90, heads_LR_r180, heads_LR_r270])

    return state_out, moves_out, heads_out


def flip_predictions_left_right(predictions: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Flip predictions left-right.

    Args:
        predictions (npt.NDArray[np.int32]): Move list

    Returns:
        npt.NDArray[np.int32]: Flipped predictions.
    """
    moves_LR = np.copy(predictions)
    moves_LR[:, 3], moves_LR[:, 1] = predictions[:, 1], predictions[:, 3]
    return moves_LR


def flip_predictions_up_down(predictions: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Flip predicitons up-down.

    Args:
        predictions (npt.NDArray[np.int32]): Predictions

    Returns:
        npt.NDArray[np.int32]: Flipped predictions.
    """
    moves_UD = np.copy(predictions)
    moves_UD[:, 0], moves_UD[:, 2] = predictions[:, 2], predictions[:, 0]
    return moves_UD


def rotate_predictions(predictions: npt.NDArray[np.int32], quads: int) -> npt.NDArray[np.int32]:
    """Rotate predictions 90 deg * quads.

    Args:
        predictions (npt.NDArray[np.int32]): Move predictions.
        quads (int): Number of quadrents to rotate.

    Returns:
        npt.NDArray[np.int32]: Rotated predictions.
    """
    return np.roll(predictions, -quads, axis=1)
