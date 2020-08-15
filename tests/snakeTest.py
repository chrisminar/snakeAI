import unittest
from snake import Snake
from snakeRL import SnakeRL
from globalVar import Globe as globe
from neuralNet import NeuralNetwork
import numpy as np
from playGames import PlayGames
from dataTrack import DataTrack

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
    self.assertEqual(s.length, 0)
    x = s.X
    y = s.Y
    l = s.length
    s.runSingle(-1,0) #move left
    self.assertEqual(s.X, x-1)
    self.assertEqual(s.Y, y)
    self.assertEqual(s.length, 0)
    x = s.X
    y = s.Y
    l = s.length
    s.runSingle(0,1) #move up
    self.assertEqual(s.X, x)
    self.assertEqual(s.Y, y+1)
    self.assertEqual(s.length, 0)
    x = s.X
    y = s.Y
    l = s.length
    s.runSingle(0,-1) #move down
    self.assertEqual(s.X, x)
    self.assertEqual(s.Y, y-1)
    self.assertEqual(s.length, 0)

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

  def test_checkGameOver_right(self):
    s = Snake ( globe.GRID_X, globe.GRID_Y )
    #head at 0,0
    s.runSingle(1,0)#head at 1,0
    self.assertFalse(s.gameover)
    s.runSingle(1,0)#head at 2,0
    self.assertFalse(s.gameover)
    s.runSingle(1,0)#head at 3,0
    self.assertFalse(s.gameover)
    s.runSingle(1,0)#head at 4,0 (dead)
    self.assertTrue(s.gameover)

  def test_checkGameOver_left(self):
    s = Snake ( globe.GRID_X, globe.GRID_Y )
    # head at 0,0
    s.runSingle(-1,0) #head at -1,0
    self.assertTrue(s.gameover)

  def test_checkGameOver_up(self):
    s = Snake ( globe.GRID_X, globe.GRID_Y )
    #head at 0,0
    s.runSingle(0,1)#head at 0,1
    self.assertFalse(s.gameover)
    s.runSingle(0,1)#head at 0,2
    self.assertFalse(s.gameover)
    s.runSingle(0,1)#head at 0,3
    self.assertFalse(s.gameover)
    s.runSingle(0,1)#head at 0,4 (dead)
    self.assertTrue(s.gameover)

  def test_checkGameOver_down(self):
    s = Snake ( globe.GRID_X, globe.GRID_Y )
    #head at 0,0
    self.assertFalse(s.gameover)
    s.runSingle(0,-1)#head at 0,-1
    self.assertTrue(s.gameover)

  def test_checkGameOver_tail(self):
    s = Snake ( globe.GRID_X, globe.GRID_Y )
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
    tr = DataTrack()
    s = SnakeRL(nn=nn, sizeX = globe.GRID_X, sizeY = globe.GRID_Y)
    g = PlayGames(tr,nn)
    direction, move_array, head = s.evaluateNextStep(g.gamestate_to_nn)
    print(direction)
    self.assertGreaterEqual(np.argmax(direction),0,'invalid direction output')
    self.assertLessEqual(np.argmax(move_array),3, 'invalid direction output')
    self.assertEqual(np.argmax(move_array),direction, 'direction doesn\'t match move array')
    self.assertEqual(head[0], 1, 'up is free')
    self.assertEqual(head[1], 1, 'right is free')
    self.assertEqual(head[2], 0, 'down is free')
    self.assertEqual(head[3], 0, 'left is free')

  def test_play(self):
    nn = NeuralNetwork()
    tr = DataTrack()
    s = SnakeRL(nn=nn, sizeX = globe.GRID_X, sizeY = globe.GRID_Y)
    g = PlayGames(tr, nn)
    s.play(g.gamestate_to_nn)
    self.assertTrue(s.gameover,False)
    self.assertGreater(len(s.moveList),0)
    print(s.moveList)
    print(s.grid)

if __name__ == '__main__':
  unittest.main()


