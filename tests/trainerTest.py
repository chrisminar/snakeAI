import unittest
from trainer import Trainer
import numpy as np
from neuralNet import NeuralNetwork
from snakeRL import SnakeRL
from globalVar import Globe as globe

class Trainer_test(unittest.TestCase):

  def test_train(self):
    self.assertEqual(0,1)

  def test_permute(self):
    nn = NeuralNetwork()
    s = SnakeRL( nn=nn, sizeX=globe.GRID_X, sizeY=globe.GRID_Y )
    s.grid[s.foodX][s.foodY] = -1
    s.foodX = globe.GRID_X + 1
    s.foodY = globe.GRID_Y + 1
    for move in range(3):
      if move == 2:
        s.foodX = s.X+1
        s.foodY = s.Y;
        s.grid[s.foodX][s.foodY] = -2
      s.runStep(1) #move right

    states = np.stack(s.stateList)
    moves  = np.stack(s.moveList)
    scores = np.full( (len(s.stateList), ), s.score )

    #lr
    stateslr = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    stateslr[0,globe.GRID_X+0,globe.GRID_Y] = 0
    stateslr[1,globe.GRID_X-1,globe.GRID_Y] = 0
    stateslr[2,globe.GRID_X-2,globe.GRID_Y] = 0
    stateslr[3,globe.GRID_X-2,globe.GRID_Y] = 1
    stateslr[3,globe.GRID_X-3,globe.GRID_Y] = 0

    moveslr = np.zeros( (4,4) )
    moveslr[0,3] = 1
    moveslr[1,3] = 1
    moveslr[2,3] = 1
    moveslr[3,3] = 1

    scoreslr = np.array([0,0,0,100])

    #ud
    statesud = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    statesud[0,globe.GRID_X+0,globe.GRID_Y] = 0
    statesud[1,globe.GRID_X+1,globe.GRID_Y] = 0
    statesud[2,globe.GRID_X+2,globe.GRID_Y] = 0
    statesud[3,globe.GRID_X+2,globe.GRID_Y] = 1
    statesud[3,globe.GRID_X+3,globe.GRID_Y] = 0

    movesud = np.zeros( (4,4) )
    movesud[0,1] = 1
    movesud[1,1] = 1
    movesud[2,1] = 1
    movesud[3,1] = 1

    scoresud = np.array([0,0,0,100])

    #90
    states90 = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    states90[0,globe.GRID_X,globe.GRID_Y+0] = 0
    states90[1,globe.GRID_X,globe.GRID_Y+1] = 0
    states90[2,globe.GRID_X,globe.GRID_Y+2] = 0
    states90[3,globe.GRID_X,globe.GRID_Y+3] = 1
    states90[3,globe.GRID_X,globe.GRID_Y+3] = 0

    moves90 = np.zeros( (4,4) )
    moves90[0,0] = 1
    moves90[1,0] = 1
    moves90[2,0] = 1
    moves90[3,0] = 1

    scores90 = np.array([0,0,0,100])

    #180
    states180 = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    states180[0,globe.GRID_X-0,globe.GRID_Y] = 0
    states180[1,globe.GRID_X-1,globe.GRID_Y] = 0
    states180[2,globe.GRID_X-2,globe.GRID_Y] = 0
    states180[3,globe.GRID_X-2,globe.GRID_Y] = 1
    states180[3,globe.GRID_X-3,globe.GRID_Y] = 0

    moves180 = np.zeros( (4,4) )
    moves180[0,3] = 1
    moves180[1,3] = 1
    moves180[2,3] = 1
    moves180[3,3] = 1

    scores180 = np.array([0,0,0,100])

    #270
    states270 = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    states270[0,globe.GRID_X,globe.GRID_Y-0] = 0
    states270[1,globe.GRID_X,globe.GRID_Y-1] = 0
    states270[2,globe.GRID_X,globe.GRID_Y-2] = 0
    states270[3,globe.GRID_X,globe.GRID_Y-3] = 1
    states270[3,globe.GRID_X,globe.GRID_Y-3] = 0

    moves270 = np.zeros( (4,4) )
    moves270[0,2] = 1
    moves270[1,2] = 1
    moves270[2,2] = 1
    moves270[3,2] = 1

    scores270 = np.array([0,0,0,100])

    #accumulate
    st,mv,sc = Trainer.permute_inputs(states,scores,moves)
    stateAll = np.stack([states, stateslr, statesUD, states90, states180, states270])
    movesAll = np.stack([moves, moveslr, movesud, moves90, moves180, moves270])
    scoresAll = np.stack([scores, scoreslr, scoresud, scores90, scores180, scores270])
    #test
    self.assertTrue(np.array_equal(sc, scoresAll))
    self.assertTrue(np.array_equal(mv, movesAll))
    self.assertTrue(np.array_equal(st, stateAll))

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


