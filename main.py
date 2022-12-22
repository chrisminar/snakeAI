"""Entrypoint to reinforcement learning."""

import logging

import absl.logging

from neural_net_improvement.neural_net_improvement_main import test_tensorboard
from training.train_snake_reinforcement_learning import TrainRL

# turns off annoying TF/keras warnings that break logger
absl.logging.set_verbosity(absl.logging.ERROR)

LOGGER = logging.getLogger("terminal")
LOGGER.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)
LOGGER.propagate = False  # prevent keras from doing annoying things with the logger

if __name__ == "__main__":
    #from training.visualize import gen_compare
    #gen_compare((40, 40, 40, 40), n=1)
    # test_tensorboard()
    trainer = TrainRL(0)
    trainer.train()
