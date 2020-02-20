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
  def __init__( self, dfTrack:dataTrack, nn:neural_network, trainingData ):
    self.td = trainingData
    self.dfDetail = dfDetail
    self.nn = nn


  def train( self, generation ):
    with Timer as t():
      #todo you are here, calc num minibatches, call training
      num_minibatch = 0
      #train
    self.dfTrack.appendTraining(t.secs, num_minibatch, td.secs/num_minibatch)

