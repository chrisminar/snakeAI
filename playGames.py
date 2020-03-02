from snakeRL import SnakeRL as snake
from dataTrack import DataTrack
from neuralNet import NeuralNetwork
import numpy as np

#####################
## self play class ##
#####################
class PlayGames():
  """description of class"""
  def __init__( self, dfTrack:DataTrack, nn:NeuralNetwork ):
    self.gamestates = []
    self.gameScore = []
    self.prediction = []
    self.gameID = []
    self.dfDetail = dfDetail
    self.nn = nn
    self.gamestate_to_nn = SelfPlay.vectorizeFunction(SelfPlay.grid_val_to_nn)

  def vectorizeFunction(functionIn):
    return np.vectorize(functionIn)

  def grid_val_to_nn(input):
    if input == -1: #empty
      out = 1
    elif input >0: #body
      out = -1
    elif input == 0: #head
      out = -2
    else: #food
      out = 2
    return out


