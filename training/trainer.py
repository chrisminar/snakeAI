"""Train neural network on best games."""
from typing import Tuple

import numpy as np
from numpy import typing as npt

from training.helper import Timer
from training.neural_net import NeuralNetwork


def train(generation: int,
          game_states: npt.NDArray[np.int32],
          heads: npt.NDArray[np.bool8],
          move_predictions: npt.NDArray[np.float32],
          verbose: int = 2,
          permute_input: bool = False) -> NeuralNetwork:
    """Permute input data and train on it.

    Args:
        generation (int): Generation number
        game_states (npt.NDArray[np.int32]): grids
        heads (npt.NDArray[np.int32]): head availiblity
        move_predictions (npt.NDArray[np.int32]): predicted moves
        permute_input (bool): If true permute inputs, will create 8x samples by rotating and flipping.

    Returns:
        (NeuralNetwork): trained neural network
    """
    neural_net = NeuralNetwork()

    with Timer(name="Training"):
        if permute_input:
            # get all permutations
            permuted_states, permuted_heads, permuted_predictions = permute_inputs(
                states=game_states, predictions=move_predictions, heads=heads)

        # train on permutations
        neural_net.train(states=permuted_states if permute_input else game_states,
                         heads=permuted_heads if permute_input else heads,
                         predictions=permuted_predictions if permute_input else move_predictions,
                         generation=generation,
                         verbose=verbose)
    return neural_net


def permute_inputs(*,
                   states: npt.NDArray[np.int32],
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
    state_lr = np.flip(states, axis=flip_axis-1)
    moves_lr = flip_predictions_left_right(heads)
    heads_lr = flip_predictions_left_right(predictions)

    # rotate lr 90
    state_lr_r90 = np.rot90(state_lr, 1, (1, 2))
    moves_lr_r90 = rotate_predictions(moves_lr, 1)
    heads_lr_r90 = rotate_predictions(heads_lr, 1)

    # rotate lr 180
    state_lr_r180 = np.rot90(state_lr, 2, (1, 2))
    moves_lr_r180 = rotate_predictions(moves_lr, 2)
    heads_lr_r180 = rotate_predictions(heads_lr, 2)

    # rotate lr 270
    state_lr_r270 = np.rot90(state_lr, 3, (1, 2))
    moves_lr_r270 = rotate_predictions(moves_lr, 3)
    heads_lr_r270 = rotate_predictions(heads_lr, 3)

    state_out = np.vstack([states, state_r90, state_r180, state_r270,
                           state_lr, state_lr_r90, state_lr_r180, state_lr_r270])
    moves_out = np.vstack([heads, moves_r90, moves_r180, moves_r270,
                           moves_lr, moves_lr_r90, moves_lr_r180, moves_lr_r270])
    heads_out = np.vstack([predictions, heads_r90, heads_r180, heads_r270,
                           heads_lr, heads_lr_r90, heads_lr_r180, heads_lr_r270])

    return state_out, moves_out, heads_out


def flip_predictions_left_right(predictions: npt.NDArray[np.number | np.bool8]) -> npt.NDArray[np.number | np.bool8]:
    """Flip predictions left-right.

    Args:
        predictions (npt.NDArray[np.int32]): Move list

    Returns:
        npt.NDArray[np.int32]: Flipped predictions.
    """
    moves_lr = np.copy(predictions)
    moves_lr[:, 3], moves_lr[:, 1] = predictions[:, 1], predictions[:, 3]
    return moves_lr


def flip_predictions_up_down(predictions: npt.NDArray[np.number | np.bool8]) -> npt.NDArray[np.number | np.bool8]:
    """Flip predicitons up-down.

    Args:
        predictions (npt.NDArray[np.int32]): Predictions

    Returns:
        npt.NDArray[np.int32]: Flipped predictions.
    """
    moves_ud = np.copy(predictions)
    moves_ud[:, 0], moves_ud[:, 2] = predictions[:, 2], predictions[:, 0]
    return moves_ud


def rotate_predictions(predictions: npt.NDArray[np.number | np.bool8], quads: int) -> npt.NDArray[np.number | np.bool8]:
    """Rotate predictions 90 deg * quads.

    Args:
        predictions (npt.NDArray[np.int32]): Move predictions.
        quads (int): Number of quadrents to rotate.

    Returns:
        npt.NDArray[np.int32]: Rotated predictions.
    """
    return np.roll(predictions, -quads, axis=1)
