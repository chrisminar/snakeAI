"""Global variables and helper functions."""
import sys
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
TIMEOUT: Final = 16

SCORE_PER_FOOD = 100


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
