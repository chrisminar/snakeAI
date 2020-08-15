import unittest
from trainer import Trainer
import numpy as np
from neuralNet import NeuralNetwork
from snakeRL import SnakeRL
from globalVar import Globe as globe
from dataTrack import DataTrack

class Trainer_test(unittest.TestCase):

  def test_train(self):
    df = DataTrack()
    nn = NeuralNetwork()
    t = Trainer(df, nn)

if __name__ == '__main__':
  unittest.main()


