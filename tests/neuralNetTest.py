import unittest

import numpy as np
import tensorflow as tf
from globalVar import Globe as globe
from neuralNet import NeuralNetwork
from tensorflow import keras


class NeuralNet_test(unittest.TestCase):
    def test_init(self):
        nn = NeuralNetwork()
        self.assertEqual(nn.model.get_layer('policy').output_shape, (None, 4))

    def test_display_model(self):
        nn = NeuralNetwork()
        nn.disp_model()

    def test_load(self):
        pass  # self.assertEqual(0,1)


if __name__ == '__main__':
    unittest.main()
