"""Global variables and helper functions."""
from __future__ import annotations

import sys
import time
from gc import get_referents
from types import FunctionType, ModuleType
from typing import Any, Final

GRID_X: Final = 4                # x grid size of snake game
GRID_Y: Final = 4                # y grid size of snake game
NUM_SELF_PLAY_GAMES: Final = 500  # number of self play games to play
NUM_PURGE: Final = 500
NUM_TRAINING_GAMES: Final = 5000  # number of self play games to train on
VALIDATION_SPLIT: Final = 0.15
EPOCH_DELTA: Final = 0.001
MOMENTUM: Final = 0.9
BATCH_SIZE: Final = 64
EPOCHS: Final = 10
MAXIMUM_MOVES_WITHOUT_EATING: Final = GRID_X * GRID_Y
MAXIMUM_TOTAL_MOVES: Final = MAXIMUM_MOVES_WITHOUT_EATING ** 2

SCORE_PER_FOOD: Final = 100  # point modification for eating food
SCORE_PER_MOVE: Final = -1  # point modificaiton for moving
SCORE_PENALTY_FOR_FAILURE: Final = -50  # point modification for dying
SCORE_FOR_GAME_WIN: Final = 1000  # get this many points for winning the game


def get_size(obj: Any) -> int:
    black_list_types = type, ModuleType, FunctionType
    """sum size of object & members."""
    if isinstance(obj, black_list_types):
        raise TypeError(
            'getsize() does not take argument of type: ' + str(type(obj)))
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


class Timer(object):
    def __init__(self, name='', verbose=False) -> None:
        self.verbose = verbose
        self.name = name

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end = time.perf_counter()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('{} elapsed time: {:0.3f} ms'.format(self.name, self.secs))
