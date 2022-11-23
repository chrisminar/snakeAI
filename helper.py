"""Global variables and helper functions."""
import sys
from gc import get_referents
from types import FunctionType, ModuleType
from typing import Any


class Globe:
    GRID_X = 4                # x grid size of snake game
    GRID_Y = 4                # y grid size of snake game
    NUM_SELF_PLAY_GAMES = 500  # number of self play games to play
    NUM_PURGE = 500
    NUM_TRAINING_GAMES = 5000  # number of self play games to train on
    VALIDATION_SPLIT = 0.15
    EPOCH_DELTA = 0.001
    MOMENTUM = 0.9
    BATCH_SIZE = 64
    EPOCHS = 10
    TIMEOUT = 16

    SCORE_PER_FOOD = 100

    def get_size(obj: Any) -> int:
        BLACKLIST = type, ModuleType, FunctionType
        """sum size of object & members."""
        if isinstance(obj, BLACKLIST):
            raise TypeError(
                'getsize() does not take argument of type: ' + str(type(obj)))
        seen_ids = set()
        size = 0
        objects = [obj]
        while objects:
            need_referents = []
            for obj in objects:
                if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    size += sys.getsizeof(obj)
                    need_referents.append(obj)
            objects = get_referents(*need_referents)
        return size
