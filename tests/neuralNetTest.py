from neuralNet import NeuralNetwork
import unittest
import numpy as np
from tensorflow import keras
from globalVar import Globe as globe


class NeuralNet_test(unittest.TestCase):
  def test_init(self):
    nn = NeuralNetwork()
    #nn.dispModel()
    self.assertEqual(0,1)

  def test_evaluate(self):
    self.assertEqual(0,1)

  def test_step_decay_schedule(self):
    self.assertEqual(0,1)

  def test_train(self):
    self.assertEqual(0,1)

  def test_conv(self):
    nn = NeuralNetwork()
    input = keras.Input( shape = ( globe.GRID_X, globe.GRID_Y, 1 ), name = 'input_game_state')
    out = nn.convBlock(0, input)
    model = keras.Model(inputs=input, outputs=out)
    print(model.layers[0].input_shape)
    print(model.layers[-1].output_shape)
    self.assertEqual(model.layers[0].input_shape, [(None, globe.GRID_X, globe.GRID_Y, 1)])
    self.assertEqual(model.layers[-1].output_shape, (None, globe.GRID_X, globe.GRID_Y, 64))

  def test_residual(self):
    self.assertEqual(0,1)

  def test_PolicyHead(self):
    self.assertEqual(0,1)

  def test_ValueHead(self):
    self.assertEqual(0,1)

  def test_displayModel(self):
    self.assertEqual(0,1)

  def test_saveload(self):
    self.assertEqual(0,1)

if __name__ == '__main__':
  unittest.main()
