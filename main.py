"""Entrypoint to reinforcement learning."""

from train_snake_reinforcement_learning import TrainRL

if __name__ == "__main__":
    trainer = TrainRL()
    trainer.train()
