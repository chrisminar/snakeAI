import unittest
from gameState import GameState
import numpy as np
from globalVar import Globe as globe

class GameState_test(unittest.TestCase):
  def test_init1(self):
    gs = GameState()
    self.assertEqual(gs.grid.shape, (globe.GRID_X,globe.GRID_Y))
    self.assertEqual(gs.grid.size, globe.GRID_X*globe.GRID_Y)

  def test_init2(self):
    gs = GameState(np.zeros((8,8)))
    self.assertEqual(gs.grid.shape, (8,8))
    self.assertEqual(gs.grid.size, 64)

if __name__ == '__main__':
  unittest.main()
