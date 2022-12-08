"""Entrypoint to reinforcement learning."""

import logging

import absl.logging

from training.train_snake_reinforcement_learning import TrainRL

# turns off annoying TF/keras warnings that break logger
absl.logging.set_verbosity(absl.logging.ERROR)

LOGGER = logging.getLogger("terminal")
LOGGER.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)
LOGGER.propagate = False  # prevent keras from doing annoying things with the logger


if __name__ == "__main__":
    trainer = TrainRL()
    trainer.train()
