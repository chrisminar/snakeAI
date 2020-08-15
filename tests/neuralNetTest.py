from neuralNet import NeuralNetwork
import unittest
import numpy as np
from tensorflow import keras
from globalVar import Globe as globe
from tensorflow.python.keras import layers
import tensorflow as tf

class NeuralNet_test(unittest.TestCase):
  def test_init(self):
    nn = NeuralNetwork()
    self.assertEqual(nn.model.get_layer('policy').output_shape, (None, 4) )

  def test_displayModel(self):
    nn=NeuralNetwork()
    nn.dispModel()

  def test_load(self):
    pass #self.assertEqual(0,1)

if __name__ == '__main__':
  unittest.main()
