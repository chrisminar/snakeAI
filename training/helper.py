"""Global variables and helper functions."""
from __future__ import annotations

import logging
import sys
import time
from enum import IntEnum
from gc import get_referents
from types import FunctionType, ModuleType
from typing import Any, Final, Optional, Sequence, Tuple, Type

import numpy as np
from matplotlib import pyplot as plt

LOGGER = logging.getLogger("terminal")

GRID_X: Final = 4                # x grid size of snake game
GRID_Y: Final = 4                # y grid size of snake game
NUM_SELF_PLAY_GAMES: Final = 500  # number of self play games to play
NUM_GAMES_PER_BATCH: Final = 10000
NUM_TRAINING_GAMES: Final = 10000  # number of self play games to train on
VALIDATION_SPLIT: Final = 0.15  # fraction of data to use for validation
EPOCH_DELTA: Final = 0.001
MOMENTUM: Final = 0.25
LEARNING_RATE: Final = 0.001
BATCH_SIZE: Final = 64
EPOCHS: Final = 100
MAXIMUM_MOVES_WITHOUT_EATING: Final = GRID_X * GRID_Y * 2
MAXIMUM_TOTAL_MOVES: Final = MAXIMUM_MOVES_WITHOUT_EATING ** 2

SCORE_PER_FOOD: Final = 100  # point modification for eating food
SCORE_PER_MOVE: Final = -1  # point modificaiton for moving
SCORE_PENALTY_FOR_FAILURE: Final = -50  # point modification for dying
SCORE_FOR_GAME_WIN: Final = 100  # get this many points for winning the game

# Odds of taking a random move to explore while training
EXPLORATORY_MOVE_FRACTION: Final = 0.1

SAVE_INTERVAL: Final = 10  # save every x generations

USE_EXPLORATION_CUTOFF: Final = 1000


class GridEnum(IntEnum):
    """Grid cell meanings."""
    FOOD = -2  # grid value that represents food
    EMPTY = -1  # grid value that represents an empty space
    HEAD = 0  # grid value that represents the head
    BODY = 1  # grid value that represents the body


class Direction(IntEnum):
    """2d direction enumerators."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def to_x_y(self) -> Tuple[int, int]:
        """Convert name to x,y grid direction.

        Returns:
            Tuple[int,int]: x and y travel direction.
        """
        return _dir_to_x_y[self]


_dir_to_x_y = {Direction.UP: (0, -1),
               Direction.RIGHT: (1, 0),
               Direction.DOWN: (0, 1),
               Direction.LEFT: (-1, 0)}


class PreProcessedGrid(IntEnum):
    """Pre processed grid values."""
    SNAKE = 1
    EMPTY = 0
    FOOD = -1


def gen_histogram(*, scores: Sequence[np.int32], generation: int) -> None:
    """Plot histogram of game scores.

    Args:
        unique (npt.NDArray[np.int32]): Game scores
        generation (int): Most recent generation.
    """
    fig = plt.figure()
    plt.title(f'Generation {generation}')
    plt.hist(scores, bins=np.arange(start=-SCORE_PER_FOOD, stop=(
        GRID_X*GRID_Y+1)*SCORE_PER_FOOD, step=SCORE_PER_FOOD))
    plt.yscale('log', nonpositive='clip')
    fig.savefig(f'media/hists/generation_{generation}.png')
    plt.close()


def get_perf(scores, ids, gen, plot: bool = True) -> float:
    """Access performance.

    Args:
        scores (_type_): Game scores
        ids (_type_): game ids
        gen (_type_): Generation
        plot (bool, optional): Should it be plotted. Defaults to True.

    Returns:
        float: mean value
    """
    _, idx = np.unique(ids, return_index=True)
    if plot:
        gen_histogram(scores=scores[idx], generation=gen)
    return np.mean(scores[idx])


def get_size(obj: Any) -> int:
    """Size of object.

    Args:
        obj (Any): any object

    Retruns:
        (int): size of object
    """
    black_list_types = type, ModuleType, FunctionType
    if isinstance(obj, black_list_types):
        raise TypeError(
            f'getsize() does not take argument of type: {type(obj)}')
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, black_list_types) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


class Timer:
    """Timing context manager."""

    def __init__(self, name: str = '') -> None:
        """Initialize timer.

        Args:
            name (str, optional): Name of timer. Defaults to ''.
            verbose (bool, optional): Should the timer print on exit. Defaults to False.
        """
        self.name = name
        self.start = 0.
        self.end = 0.
        self.secs = 0.
        self.msecs = 0.

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Any) -> None:
        self.end = time.perf_counter()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        LOGGER.info('%s elapsed time %03f', self.name, self.secs)
