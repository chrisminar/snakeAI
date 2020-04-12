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

  def test_compile_fit_evaluate_step_decay_schedule(self):
    n= 100
    trainSet = np.random.randint(3, size=(n,10,10,1))
    policy = np.random.randint(1, size=(n,4))

    nn=NeuralNetwork()
    nn.train(trainSet, policy, 0)
    nn.evaluate(np.zeros((10,10)))
    #should put some asserts here

  def test_conv(self):
    input = keras.Input( shape = ( globe.GRID_X, globe.GRID_Y, 1 ), name = 'input_game_state')
    out = NeuralNetwork.convBlock(0, input)
    model = keras.Model(inputs=input, outputs=out)
    print(model.layers[0].input_shape)
    print(model.layers[-1].output_shape)
    self.assertEqual(model.layers[0].input_shape, [(None, globe.GRID_X, globe.GRID_Y, 1)])
    self.assertEqual(model.layers[-1].output_shape, (None, globe.GRID_X, globe.GRID_Y, 64))

  def test_residual(self):
    input = keras.Input( shape = ( globe.GRID_X, globe.GRID_Y, 64 ), name = 'input_residual')
    out = NeuralNetwork.residualBlock(0, input)
    model = keras.Model(inputs=input, outputs=out)
    print(model.layers[0].input_shape)
    print(model.layers[-1].output_shape)
    self.assertEqual(model.layers[0].input_shape,  [(None, globe.GRID_X, globe.GRID_Y, 64)])
    self.assertEqual(model.layers[-1].output_shape, (None, globe.GRID_X, globe.GRID_Y, 64))

  def test_PolicyHead(self):
    input = keras.Input( shape = ( globe.GRID_X, globe.GRID_Y, 64 ), name = 'input_policyhead')
    out = NeuralNetwork.policyHead(input)
    model = keras.Model(inputs=input, outputs=out)
    print(model.layers[0].input_shape)
    print(model.layers[-1].output_shape)
    self.assertEqual(model.layers[0].input_shape,  [(None, globe.GRID_X, globe.GRID_Y, 64)])
    self.assertEqual(model.layers[-1].output_shape, (None, 4))

  def test_displayModel(self):
    nn=NeuralNetwork()
    nn.dispModel()

  def test_load(self):
    pass #self.assertEqual(0,1)

if __name__ == '__main__':
  unittest.main()
