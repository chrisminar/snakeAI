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
    self.dfTrack = dfTrack
    self.nn = nn
    self.gamestate_to_nn = PlayGames.vectorizeFunction(PlayGames.grid_val_to_nn)

  def vectorizeFunction(functionIn):
    return np.vectorize(functionIn)

  def grid_val_to_nn(input):
    """Convert input snake grid value to nn value"""
    # empty -> 1
    # head -> 0
    # body = negative
    #food = 2
    return -input