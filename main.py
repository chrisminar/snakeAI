"""Entrypoint to reinforcement learning."""

import logging

from training.train_snake_reinforcement_learning import TrainRL

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

if __name__ == "__main__":
    trainer = TrainRL()
    trainer.train()
