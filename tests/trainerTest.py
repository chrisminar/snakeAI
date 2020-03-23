import unittest
from trainer import Trainer
import numpy as np


class Trainer_test(unittest.TestCase):

  def test_train(self):
    self.assertEqual(0,1)

  def test_permute(self):
    self.assertEqual(0,1)

  def test_flipLR(self):
    input = np.array([[1,0,0,0],#up
                      [0,0,1,0],#down
                      [0,0,0,1],#left
                      [0,1,0,0]])#right
    output = np.array([[1,0,0,0],#up
                       [0,0,1,0],#down
                       [0,1,0,0],#left
                       [0,0,0,1]])#right

    self.assertTrue(np.array_equal(Trainer.flipMoveLR(input),output),'congregate failed')

  def test_flipUD(self):
    input = np.array([[1,0,0,0],#up
                      [0,0,1,0],#down
                      [0,0,0,1],#left
                      [0,1,0,0]])#right
    output = np.array([[0,0,1,0],#up
                       [1,0,0,0],#down
                       [0,0,0,1],#left
                       [0,1,0,0]])#right

    self.assertTrue(np.array_equal(Trainer.flipMoveUD(input),output),'congregate failed')

  def test_rotate(self):
    input = np.array([[1,0,0,0],#up
                      [0,0,1,0],#down
                      [0,0,0,1],#left
                      [0,1,0,0]])#right
    output = np.array([[0,0,0,1],#up
                       [0,1,0,0],#down
                       [0,0,1,0],#left
                       [1,0,0,0]])#right

    self.assertTrue(np.array_equal(Trainer.rotateMoves(input,1),output),'rotate 90')
    input = np.array([[1,0,0,0],#up
                      [0,0,1,0],#down
                      [0,0,0,1],#left
                      [0,1,0,0]])#right
    output = np.array([[0,0,1,0],#up
                       [1,0,0,0],#down
                       [0,1,0,0],#left
                       [0,0,0,1]])#right

    self.assertTrue(np.array_equal(Trainer.rotateMoves(input,2),output),'rotate 180')

    input = np.array([[1,0,0,0],#up
                      [0,0,1,0],#down
                      [0,0,0,1],#left
                      [0,1,0,0]])#right
    output = np.array([[0,1,0,0],#up
                       [0,0,0,1],#down
                       [1,0,0,0],#left
                       [0,0,1,0]])#right

    self.assertTrue(np.array_equal(Trainer.rotateMoves(input,3),output),'rotate 270')
if __name__ == '__main__':
  unittest.main()

