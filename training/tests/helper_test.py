"""Test helper."""

import time

import pytest

from training.helper import Timer


def test_timer() -> None:
    """Test timer context manager works."""
    with Timer('Timer test') as timer:
        time.sleep(0.1)
    assert pytest.approx(timer.secs, abs=1e-2) == 0.1
