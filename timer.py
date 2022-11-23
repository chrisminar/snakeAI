from __future__ import annotations

import time


class Timer(object):
    def __init__(self, name='', verbose=False) -> None:
        self.verbose = verbose
        self.name = name

    def __enter__(self) -> Timer:
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('{} elapsed time: {:0.3f} ms'.format(self.name, self.secs))
