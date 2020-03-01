from globalVar import Globe as globe
from gameState import GameState
from gameState import Game
from timer import Timer
from dataTrack import DataTrack
from neuralNet import NeuralNetwork


###################
## trainer class ##
###################
class Trainer():
  """description of class"""
  def __init__( self, dfTrack:DataTrack, nn:NeuralNetwork ):
    self.td = trainingData
    self.dfTrack = dfTrack
    self.nn = nn

  def train( self, generation:int, inputs, scores, move_predictions ):
    with Timer() as t:
      self.nn.train( inputs, scores, move_predictions, generation )
      num_minibatch = move_predictions.shape[0]
    self.dfTrack.appendTraining( t.secs, num_minibatch, td.secs/num_minibatch )

