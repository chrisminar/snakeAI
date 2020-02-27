#todo, convert and store gamelist as nn state
from globalVar import Globe as globe
from gameState import GameState
from gameState import Game
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
    self.gameID = []
    self.dfDetail = dfDetail
    self.nn = nn

  def playGames( self, generation:int, startID:int ):
    for i in range(globe.NUM_TRAINING_GAMES):
      with Timer() as t:
        g = Game(nn=self.nn, sizeX=globe.GRID_X, sizeY=globe.GRID_Y)
        g.play()
        gamestates.append(np.stack(g.stateList))
        gameScore.append( np.full( (len(g.stateList), ), g.score ) )
        gameID.append( np.full( (len(g.stateList), ), startID+i ) )
      self.dfTrack.appendSelfPlayDetail(t.secs, g.score, generation, i)
    self.dfTrack.appendSelfPlayBroad(self.dfTrack.self_play_detail.loc[generation,'time'].sum(), self.dfTrack.self_play_detail.loc[generation,'score'].mean())

    return np.concatenate(self.gamestates), np.concatenate(self.gameScore), np.concatenate(self.gameID)