import unittest
from trainer import Trainer
import numpy as np
from neuralNet import NeuralNetwork
from snakeRL import SnakeRL
from globalVar import Globe as globe

class Trainer_test(unittest.TestCase):

  def test_train(self):
    nn = NeuralNetwork()
    t = Trainer(nn)

if __name__ == '__main__':
  unittest.main()


