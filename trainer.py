from globalVar import Globe as globe
from gameState import GameState
from timer import Timer
from dataTrack import DataTrack
from neuralNet import NeuralNetwork
import numpy as np


###################
## trainer class ##
###################
class Trainer():
  """description of class"""
  def __init__( self, dfTrack:DataTrack, nn:NeuralNetwork ):
    self.dfTrack = dfTrack
    self.nn = nn

  def train( self, generation:int, inputs, scores, move_predictions, meanScore:float):
    with Timer() as t:
      #take top x scores from those games
      idx = np.nonzero(scores>meanScore)

      #get all permutations
      statesP, movesP = Trainer.permute_inputs(inputs[idx], move_predictions[idx])
      
      #train on them
      self.nn.train( np.reshape(statesP,(statesP.shape[0], statesP.shape[1], statesP.shape[1], 1)), movesP, generation )
      num_minibatch = statesP.shape[0]/globe.BATCH_SIZE
    self.dfTrack.appendTraining( t.secs, num_minibatch, t.secs/num_minibatch )

  def permute_inputs(states, moves):
    flipAxis = len(states.shape)-1

    #rotate 90
    stateR90 = np.rot90(states,1,(1,2))
    movesR90 = Trainer.rotateMoves(moves, 1)

    #rotate 180
    stateR180 = np.rot90(states,2,(1,2))
    movesR180 = Trainer.rotateMoves(moves, 2)

    #rotate 270
    stateR270 = np.rot90(states,3,(1,2))
    movesR270 = Trainer.rotateMoves(moves, 3)

    #flip left - right
    stateLR = np.flip(states, axis=flipAxis-1)
    movesLR = Trainer.flipMoveLR(moves)

    #rotate lr 90
    stateLRR90 = np.rot90(stateLR,1,(1,2))
    movesLRR90 = Trainer.rotateMoves(movesLR, 1)

    #rotate lr 180
    stateLRR180 = np.rot90(stateLR,2,(1,2))
    movesLRR180 = Trainer.rotateMoves(movesLR, 2)

    #rotate lr 270
    stateLRR270 = np.rot90(stateLR,3,(1,2))
    movesLRR270 = Trainer.rotateMoves(movesLR, 3)


    stateOut = np.vstack([states, stateR90, stateR180, stateR270, stateLR, stateLRR90, stateLRR180, stateLRR270])
    movesOut = np.vstack([moves, movesR90, movesR180, movesR270, movesLR, movesLRR90, movesLRR180, movesLRR270])

    return stateOut, movesOut

  def flipMoveLR(moves):
    movesLR = np.copy(moves)
    movesLR[:,3], movesLR[:,1] = moves[:,1], moves[:,3]
    return movesLR

  def flipMoveUD(moves):
    movesUD = np.copy(moves)
    movesUD[:,0], movesUD[:,2] = moves[:,2], moves[:,0]
    return movesUD

  def rotateMoves(moves, quads:int):
    return np.roll(moves, -quads, axis=1)