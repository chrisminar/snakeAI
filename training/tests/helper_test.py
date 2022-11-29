"""Test helper."""

import time

import pytest

from helper import Timer


def test_timer() -> None:
    """Test timer context manager works."""
    with Timer('Timer test') as timer:
        time.sleep(0.1)
    assert pytest.approx(timer.secs) == 0.1
