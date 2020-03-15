import unittest
from snake import Snake
from snakeRL import SnakeRL
from globalVar import Globe as globe
import numpy as np

class Snake_test(unittest.TestCase):
  def test_init(self):
    s = Snake(globe.GRID_X,globe.GRID_Y)
    self.assertEqual(s.gameover,False)
    self.assertEqual(s.grid.shape, np.zeros((globe.GRID_X,globe.GRID_Y)).shape)

  def test_RunSingle_no_food(self):
    s = Snake( globe.GRID_X, globe.GRID_Y )
    x = s.X
    y = s.Y
    l = s.length
    s.runSingle(1,0) #move right
    self.assertEqual(s.X, x+1)
    self.assertEqual(s.Y, y)
    self.assertEqual(s.length, l)
    x = s.X
    y = s.Y
    l = s.length
    s.runSingle(-1,0) #move left
    self.assertEqual(s.X, x-1)
    self.assertEqual(s.Y, y)
    self.assertEqual(s.length, l)
    x = s.X
    y = s.Y
    l = s.length
    s.runSingle(0,1) #move up
    self.assertEqual(s.X, x)
    self.assertEqual(s.Y, y+1)
    self.assertEqual(s.length, l)
    x = s.X
    y = s.Y
    l = s.length
    s.runSingle(0,-1) #move down
    self.assertEqual(s.X, x)
    self.assertEqual(s.Y, y-1)
    self.assertEqual(s.length, l)

  def test_RunSingle_with_food(self):
    s = Snake( globe.GRID_X, globe.GRID_Y )
    x = s.X
    y = s.Y
    l = s.length
    score = s.score
    s.foodX = x+1
    s.foodY = y;
    s.grid[s.foodX][s.foodY] = -2
    s.runSingle(1,0) #move right
    self.assertEqual(s.grid[x][y], l+1)
    self.assertGreater(s.score, score)
    self.assertEqual(s.length, l+1)

  def test_spawnFood(self):
    s = Snake( globe.GRID_X, globe.GRID_Y )
    i = np.zeros((10,))
    j = np.zeros((10,))
    for food in range(10):
      i[food], j[food] = s.spawn_food()
    self.assertEqual(np.amax(s.grid), 0) #check no food
    self.assertGreater(np.std(i), 0)
    self.assertGreater(np.std(j), 0)

  def test_checkGameOver(self):
    self.assertEqual(0,1)

class SnakeRL_test(unittest.TestCase):
  def test_init(self):
    self.assertEqual(0,1)

  def test_runStep(self):
    self.assertEqual(0,1)

  def test_evaluateNext(self):
    self.assertEqual(0,1)

  def test_play(self):
    self.assertEqual(0,1)

if __name__ == '__main__':
  unittest.main()


