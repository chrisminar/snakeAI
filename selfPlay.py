#todo, convert and store gamelist as nn state
from globalVar import Globe as globe
from gameState import GameState
from snakeRL import SnakeRL as snake
from timer import Timer
from dataTrack import DataTrack
from neuralNet import NeuralNetwork
import numpy as np

#####################
## self play class ##
#####################
class SelfPlay():
  """description of class"""
  def __init__( self, dfTrack:DataTrack, nn:NeuralNetwork ):
    self.gamestates = []
    self.gameScore = []
    self.prediction = []
    self.gameID = []
    self.dfDetail = dfDetail
    self.nn = nn
    self.gamestate_to_nn = SelfPlay.vectorizeFunction(SelfPlay.grid_val_to_nn)

  def playGames( self, generation:int, startID:int ):
    for i in range(globe.NUM_TRAINING_GAMES):
      with Timer() as t:
        g = snake(nn=self.nn, sizeX=globe.GRID_X, sizeY=globe.GRID_Y)
        g.play()
        self.gamestates.append( np.stack(g.stateList))
        self.gameScore.append(  np.full( (len(g.stateList), ), g.score ) )
        self.gameID.append(     np.full( (len(g.stateList), ), startID+i ) )
        self.prediction.append( np.array(g.moveList))
      self.dfTrack.appendSelfPlayDetail(t.secs, g.score, generation, i)
    self.dfTrack.appendSelfPlayBroad(self.dfTrack.self_play_detail.loc[generation,'time'].sum(), self.dfTrack.self_play_detail.loc[generation,'score'].mean())

    return self.gamestate_to_nn(np.concatenate(self.gamestates)), np.concatenate(self.gameScore), np.concatenate(self.gameID), np.concatenate(self.prediction)

  def vectorizeFunction(functionIn):
    return np.vectorize(functionIn)

  def grid_val_to_nn(input):
    if input == -1:
      out = 1
    elif input >0:
      out = -1
    elif input == 0:
      out = -2
    else:
      out = 2
    return out

