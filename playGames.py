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
    self.prediction = []
    self.gameId = []
    self.scores = []
    self.heads = []
    self.dfTrack = dfTrack
    self.nn = nn
    self.gamestate_to_nn = PlayGames.vectorizeFunction(PlayGames.grid_val_to_nn)

  def vectorizeFunction(functionIn):
    return np.vectorize(functionIn)

  def grid_val_to_nn(input):
    """Convert input snake grid value to nn value"""
    if input == -1: # empty -1 -> 0
      return 0
    elif input == -2: #food -2 -> -1
      return -1
    else: # head 0 -> 1, body positive -> 1
      return 1