import unittest

import numpy as np
from helper import Globe as globe
from neural_net import NeuralNetwork
from snake_reinforcement_learning import SnakeRL
from trainer import Trainer


class Trainer_test(unittest.TestCase):

    def test_train(self):
        nn = NeuralNetwork()
        t = Trainer(nn)


if __name__ == '__main__':
    unittest.main()
