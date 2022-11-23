import unittest

from helper import Globe as globe
from neural_net import NeuralNetwork
from trainer import Trainer


class Trainer_test(unittest.TestCase):

    def test_train(self):
        nn = NeuralNetwork()
        t = Trainer(nn)


if __name__ == '__main__':
    unittest.main()
