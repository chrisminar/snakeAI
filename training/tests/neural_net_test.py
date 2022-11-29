"""Test neural network."""

import pytest

from neural_net import NeuralNetwork


def test_init() -> None:
    """Initialize neural net and check output layer shape."""
    neural_net = NeuralNetwork()
    assert neural_net.model.get_layer('policy').output_shape, (None, 4)


@pytest.mark.skip()
def test_display_model() -> None:
    """Test model display."""
    neural_network = NeuralNetwork()
    neural_network.disp_model()


@pytest.mark.skip()
def test_load():
    """To do."""
    raise NotImplementedError
