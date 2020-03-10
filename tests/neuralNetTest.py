from neuralNet import NeuralNetwork
import unittest
import numpy as np
from tensorflow import keras
from globalVar import Globe as globe


class NeuralNet_test(unittest.TestCase):
  def test_init(self):
    nn = NeuralNetwork()
    self.assertEqual(nn.model.get_layer('value').output_shape, (None, 1) )
    self.assertEqual(nn.model.get_layer('policy').output_shape, (None, 4) )
    self.assertEqual(0,1)
    #check weights

  def test_evaluate(self):
    self.assertEqual(0,1)

  def test_step_decay_schedule(self):
    f = NeuralNetwork.step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)
    #u r here
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
    nn = NeuralNetwork()
    input = keras.Input( shape = ( globe.GRID_X, globe.GRID_Y, 64 ), name = 'input_residual')
    out = nn.residualBlock(0, input)
    model = keras.Model(inputs=input, outputs=out)
    print(model.layers[0].input_shape)
    print(model.layers[-1].output_shape)
    self.assertEqual(model.layers[0].input_shape,  [(None, globe.GRID_X, globe.GRID_Y, 64)])
    self.assertEqual(model.layers[-1].output_shape, (None, globe.GRID_X, globe.GRID_Y, 64))

  def test_PolicyHead(self):
    nn = NeuralNetwork()
    input = keras.Input( shape = ( globe.GRID_X, globe.GRID_Y, 64 ), name = 'input_policyhead')
    out = nn.policyHead(input)
    model = keras.Model(inputs=input, outputs=out)
    print(model.layers[0].input_shape)
    print(model.layers[-1].output_shape)
    self.assertEqual(model.layers[0].input_shape,  [(None, globe.GRID_X, globe.GRID_Y, 64)])
    self.assertEqual(model.layers[-1].output_shape, (None, 4))

  def test_ValueHead(self):
    nn = NeuralNetwork()
    input = keras.Input( shape = ( globe.GRID_X, globe.GRID_Y, 64 ), name = 'input_valuehead')
    out = nn.valueHead(input)
    model = keras.Model(inputs=input, outputs=out)
    print(model.layers[0].input_shape)
    print(model.layers[-1].output_shape)
    self.assertEqual(model.layers[0].input_shape,  [(None, globe.GRID_X, globe.GRID_Y, 64)])
    self.assertEqual(model.layers[-1].output_shape, (None, 1))

  def test_displayModel(self):
    self.assertEqual(0,1)

  def test_saveload(self):
    self.assertEqual(0,1)

if __name__ == '__main__':
  unittest.main()
