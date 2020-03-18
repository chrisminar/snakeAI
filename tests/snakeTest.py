import unittest
from snake import Snake
from snakeRL import SnakeRL
from globalVar import Globe as globe
from neuralNet import NeuralNetwork
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

  def test_checkGameOver_sides(self):
    #die right
    s = Snake ( 8, 8 )
    s.runSingle(1,0)#head at 5,4
    self.assertFalse(s.gameover)
    s.runSingle(1,0)#head at 6,4
    self.assertFalse(s.gameover)
    s.runSingle(1,0)#head at 7,4
    self.assertFalse(s.gameover)
    s.runSingle(1,0)#head at 8,5 (dead)
    self.assertTrue(s.gameover)

    #die left
    s = Snake ( 8, 8 )
    s.runSingle(-1,0)#head at 3,4
    self.assertFalse(s.gameover)
    s.runSingle(-1,0)#head at 2,4
    self.assertFalse(s.gameover)
    s.runSingle(-1,0)#head at 1,4
    self.assertFalse(s.gameover)
    s.runSingle(-1,0)#head at 0,4
    self.assertFalse(s.gameover)
    s.runSingle(-1,0)#head at -1,5 (dead)
    self.assertTrue(s.gameover)

    #die up
    s = Snake ( 8, 8 )
    s.runSingle(0,1)#head at 4,5
    self.assertFalse(s.gameover)
    s.runSingle(0,1)#head at 4,6
    self.assertFalse(s.gameover)
    s.runSingle(0,1)#head at 4,7
    self.assertFalse(s.gameover)
    s.runSingle(0,1)#head at 4,8 (dead)
    self.assertTrue(s.gameover)

    #die down
    s = Snake ( 8, 8 )
    s.runSingle(0,-1)#head at 4,3
    self.assertFalse(s.gameover)
    s.runSingle(0,-1)#head at 4,2
    self.assertFalse(s.gameover)
    s.runSingle(0,-1)#head at 4,1
    self.assertFalse(s.gameover)
    s.runSingle(0,-1)#head at 4,0
    self.assertFalse(s.gameover)
    s.runSingle(0,-1)#head at 4,-1 (dead)
    self.assertTrue(s.gameover)

  def test_checkGameOver_tail(self):
    s = Snake ( 8, 8 )
    s.grid[3][4] = -2 #place food at (3,4)
    s.foodX,s.foodY = 3,4
    s.runSingle(-1,0) #move left and eat
    s.grid[3][3] = -2 #place food at (3,3)
    s.foodX,s.foodY = 3,3
    s.runSingle(0,-1) #move down and eat
    s.grid[4][3] = -2 #place food at (4,3)
    s.foodX,s.foodY = 4,3
    s.runSingle(1,0) #move right and eat
    s.runSingle(0,1) #move up into old tail spot
    self.assertFalse(s.gameover)
    s.runSingle(0,-1)#move down into tail
    self.assertTrue(s.gameover)

class SnakeRL_test(unittest.TestCase):
  def test_init(self):
    nn = NeuralNetwork()
    s = SnakeRL(nn=nn, sizeX = 8, sizeY = 8)
    self.assertTrue(hasattr(s,'nn'))

  def test_Runstep_no_food(self):
    nn = NeuralNetwork()
    s = SnakeRL( nn=nn, sizeX=globe.GRID_X, sizeY=globe.GRID_Y )
    x = s.X
    y = s.Y
    l = s.length
    s.runStep(1) #move right
    self.assertEqual(s.X, x+1)
    self.assertEqual(s.Y, y)
    self.assertEqual(s.length, l)
    x = s.X
    y = s.Y
    l = s.length
    s.runStep(3) #move left
    self.assertEqual(s.X, x-1)
    self.assertEqual(s.Y, y)
    self.assertEqual(s.length, l)
    x = s.X
    y = s.Y
    l = s.length
    s.runStep(2) #move up (up on grid, down if looking at it)
    self.assertEqual(s.X, x)
    self.assertEqual(s.Y, y+1)
    self.assertEqual(s.length, l)
    x = s.X
    y = s.Y
    l = s.length
    s.runStep(0) #move down (down on grid, up i flooking at it)
    self.assertEqual(s.X, x)
    self.assertEqual(s.Y, y-1)
    self.assertEqual(s.length, l)

  def test_Runstep_with_food(self):
    nn = NeuralNetwork()
    s = SnakeRL( nn=nn, sizeX=globe.GRID_X, sizeY=globe.GRID_Y )
    x = s.X
    y = s.Y
    l = s.length
    score = s.score
    s.foodX = x+1
    s.foodY = y;
    s.grid[s.foodX][s.foodY] = -2
    s.runStep(1) #move right
    self.assertEqual(s.grid[x][y], l+1)
    self.assertGreater(s.score, score)
    self.assertEqual(s.length, l+1)

  def test_evaluateNext(self):
    nn = NeuralNetwork()
    s = SnakeRL(nn=nn, sizeX = 10, sizeY = 10)
    direction = s.evaluateNextStep()
    print(direction)
    self.assertGreaterEqual(direction,0)
    self.assertLessEqual(direction,3)

  def test_play(self):
    nn = NeuralNetwork()
    s = SnakeRL(nn=nn, sizeX = 10, sizeY = 10)
    s.play()
    self.assertTrue(s.gameover,False)
    self.assertGreater(len(s.moveList),0)
    print(s.moveList)
    print(s.grid)

if __name__ == '__main__':
  unittest.main()


