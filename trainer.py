from globalVar import globe
from game_state import game_state
from game_state import game
from timer import Timer
from dataTrack import dataTrack
from neuralNet import neural_network


###################
## trainer class ##
###################
class trainerClass():
  """description of class"""
  def __init__( self, dfTrack:dataTrack, nn:neural_network):
    self.td = trainingData
    self.dfTrack = dfTrack
    self.nn = nn

  def train( self, generation, inputs, scores, move_predictions):
    with Timer as t():
      self.nn.train( inputs, scores, move_predictions, generation)
      num_minibatch = move_predictions.shape[0]
    self.dfTrack.appendTraining(t.secs, num_minibatch, td.secs/num_minibatch)

