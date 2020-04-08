import unittest
from trainer import Trainer
import numpy as np
from neuralNet import NeuralNetwork
from snakeRL import SnakeRL
from globalVar import Globe as globe
from dataTrack import DataTrack

class Trainer_test(unittest.TestCase):

  def test_train(self):
    df = DataTrack()
    nn = NeuralNetwork()
    t = Trainer(df, nn)

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
      m = [0,1,0,0]
      s.moveList.append(m)
      s.stateList.append(np.copy(s.grid))
      s.runStep(1) #move right

    s.moveList.append(m)
    s.stateList.append(np.copy(s.grid))
    m = [0,1,0,0]

    states = np.stack(s.stateList)
    moves  = np.stack(s.moveList)
    scores = np.full( (len(s.stateList), ), s.score )

    gxStart = int(globe.GRID_X/2)
    gyStart = int(globe.GRID_Y/2)

    #90
    states90 = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    states90[0,gxStart-1,gyStart+0] = 0
    states90[1,gxStart-1,gyStart+1] = 0
    states90[2,gxStart-1,gyStart+2] = 0
    states90[3,gxStart-1,gyStart+2] = 1
    states90[3,gxStart-1,gyStart+3] = 0

    moves90 = np.zeros( (4,4) )
    moves90[0,0] = 1
    moves90[1,0] = 1
    moves90[2,0] = 1
    moves90[3,0] = 1

    #180
    states180 = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    states180[0,gxStart-1, gyStart-1] = 0
    states180[1,gxStart-2, gyStart-1] = 0
    states180[2,gxStart-3, gyStart-1] = 0
    states180[3,gxStart-3, gyStart-1] = 1
    states180[3,gxStart-4, gyStart-1] = 0

    moves180 = np.zeros( (4,4) )
    moves180[0,3] = 1
    moves180[1,3] = 1
    moves180[2,3] = 1
    moves180[3,3] = 1

    #270
    states270 = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    states270[0,gxStart, gyStart -1] = 0
    states270[1,gxStart, gyStart -2] = 0
    states270[2,gxStart, gyStart -3] = 0
    states270[3,gxStart, gyStart -3] = 1
    states270[3,gxStart, gyStart -4] = 0

    moves270 = np.zeros( (4,4) )
    moves270[0,2] = 1
    moves270[1,2] = 1
    moves270[2,2] = 1
    moves270[3,2] = 1

    #lr
    stateslr = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    stateslr[0,gxStart-1,gyStart] = 0
    stateslr[1,gxStart-2,gyStart] = 0
    stateslr[2,gxStart-3,gyStart] = 0
    stateslr[3,gxStart-3,gyStart] = 1
    stateslr[3,gxStart-4,gyStart] = 0

    moveslr = np.zeros( (4,4) )
    moveslr[0,3] = 1
    moveslr[1,3] = 1
    moveslr[2,3] = 1
    moveslr[3,3] = 1

    #lr 90
    stateslr90 = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    stateslr90[0,gxStart-1,gyStart-1] = 0
    stateslr90[1,gxStart-1,gyStart-2] = 0
    stateslr90[2,gxStart-1,gyStart-3] = 0
    stateslr90[3,gxStart-1,gyStart-3] = 1
    stateslr90[3,gxStart-1,gyStart-4] = 0

    moveslr90 = np.zeros( (4,4) )
    moveslr90[0,2] = 1
    moveslr90[1,2] = 1
    moveslr90[2,2] = 1
    moveslr90[3,2] = 1

    #lr 180
    stateslr180 = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    stateslr180[0,gxStart+0,gyStart-1] = 0
    stateslr180[1,gxStart+1,gyStart-1] = 0
    stateslr180[2,gxStart+2,gyStart-1] = 0
    stateslr180[3,gxStart+2,gyStart-1] = 1
    stateslr180[3,gxStart+3,gyStart-1] = 0

    moveslr180 = np.zeros( (4,4) )
    moveslr180[0,1] = 1
    moveslr180[1,1] = 1
    moveslr180[2,1] = 1
    moveslr180[3,1] = 1

    #lr 270
    stateslr270 = np.zeros( (4,globe.GRID_X,globe.GRID_Y) ) - 1
    stateslr270[0,gxStart,gyStart+0] = 0
    stateslr270[1,gxStart,gyStart+1] = 0
    stateslr270[2,gxStart,gyStart+2] = 0
    stateslr270[3,gxStart,gyStart+2] = 1
    stateslr270[3,gxStart,gyStart+3] = 0

    moveslr270 = np.zeros( (4,4) )
    moveslr270[0,0] = 1
    moveslr270[1,0] = 1
    moveslr270[2,0] = 1
    moveslr270[3,0] = 1

    #accumulate
    states = np.where(states==-2, -1, states) #set -2s to -1s
    st,mv,sc = Trainer.permute_inputs(states,scores,moves)
    self.assertTrue(np.array_equal(st[:4,:,:],    states),      'states')
    self.assertTrue(np.array_equal(st[4:8,:,:],   states90),    'rotate90')
    self.assertTrue(np.array_equal(st[8:12,:,:],  states180),   'rotate180')
    self.assertTrue(np.array_equal(st[12:16,:,:], states270),   'rotate270')
    self.assertTrue(np.array_equal(st[16:20,:,:], stateslr),    'stateslr')
    self.assertTrue(np.array_equal(st[20:24,:,:], stateslr90),  'rotatelr90')
    self.assertTrue(np.array_equal(st[24:28,:,:], stateslr180), 'rotatelr180')
    self.assertTrue(np.array_equal(st[28:,:,:],   stateslr270), 'rotatelr270')
    stateAll = np.concatenate([states, states90, states180, states270, stateslr, stateslr90, stateslr180, stateslr270])
    movesAll = np.concatenate([moves, moves90, moves180, moves270, moveslr, moveslr90, moveslr180, moveslr270])
    scoresAll = np.concatenate([scores, scores, scores, scores, scores, scores, scores, scores])
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


