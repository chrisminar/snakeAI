# todo list
# test for timeout
# try to progressively update nn, not reinit every time?
#experiemnt with different optimizers


#################
### soon list ###
#################
# current scheme results in very low number of games played after the mean score increases.
  # should change the game trimming to be a bit more complex, e.g. always keep ~10k games and remove the worst ones

#################
### done list ###
#################
# need to take the highest valid class output, not random if highest is invalid. Can do this in the nn itself.
# change trim game list such that the highest mean score is remembered. Every time you pass Prune games below highest mean every pass.
# possible failure of trim game states after too many runs (numer of samples constantly increases)
  # need to rework. Currently takes newest game id - oldest game id. it should count the total number of game ids
# possible stagnation
  #around 450
# try training on more epochs? -- seems to stop benefitting around 2
# try init snake in middle (not 0,0)
  #moved to 1,1

###################
### FUTURE LIST ###
###################
#stuff to do someday
  # do network pruning
  # track/understand what the neural network is doing
  # gui
  # pause/play

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import typing
from typing import List,Tuple
from neuralNet import NeuralNetwork
from globalVar import Globe as globe
from timer import Timer

from selfPlay import SelfPlay
from trainer import Trainer

class TrainRL():
  def __init__(self):
    self.gameStates = np.zeros((0,globe.GRID_X,globe.GRID_Y))
    self.gameHeads = np.zeros((0,4))
    self.gameScores = np.zeros((0,))
    self.gameIDs = np.zeros((0,))
    self.moves = np.zeros((0,4))
    self.nn = NeuralNetwork()
    self.gameID = 0
    self.meanScore = 0
    self.meanScores = []
    self.gamesUsed = []
    self.maxScores = []
    return

  def train(self):
    generation = 0
    self.selfPlay(self.nn, generation, globe.NUM_TRAINING_GAMES) #do big first batch

    while 1:
      self.selfPlay(self.nn, generation)
      self.networkTrainer(generation)
      print(generation, globe.getsize(self.nn))
      self.genStatusPlot(generation)
      generation += 1
    return

  #save plot that shows status of training
  def genStatusPlot(self, generation):
    fig=plt.figure()
    plt.subplot(3,1,1)
    plt.plot(self.meanScores)
    plt.title('Mean Score')
      
    plt.subplot(3,1,2)
    plt.plot(self.maxScores)
    plt.title('Max Score')
      
    plt.subplot(3,1,3)
    plt.plot(self.gamesUsed)
    plt.title('Training Games')
      
    plt.suptitle('Generation: {}'.format(generation ))

    fig.savefig('mostrecent.png')
    plt.close()
    return

  #operates on:
    #neural network
  #outputs:
    #2000 game outputs
  def selfPlay(self, nn:NeuralNetwork, generation:int, num_games:int=globe.NUM_SELF_PLAY_GAMES ):
    spc = SelfPlay(nn)
    states, heads, scores, ids, moves = spc.playGames(generation, self.gameID, num_games)
    self.gameID += globe.NUM_SELF_PLAY_GAMES
    print( 'Moves in this training set:  Up: ', np.sum(moves[:,0]), ', Right: ', np.sum(moves[:,1]), ', Down: ', np.sum(moves[:,2]), ', Left: ', np.sum(moves[:,3]))
    self.addGamesToList(states, heads, scores, ids, moves)
    self.trimGameList(generation)
    return

  #operates on:
    # last 10000 games of self play
  #outputs:
    #new neural network
  def networkTrainer(self, generation:int):
    del self.nn
    self.nn = NeuralNetwork()
    trn = Trainer(self.nn)
    trn.train(generation, self.gameStates, self.gameHeads, self.moves)
    return

  def addGamesToList(self, states, heads, scores, ids, moves):
    self.gameStates = np.concatenate((self.gameStates, states))
    self.gameHeads  = np.concatenate((self.gameHeads, heads))
    self.gameScores = np.concatenate((self.gameScores, scores))
    self.gameIDs    = np.concatenate((self.gameIDs, ids))
    self.moves      = np.concatenate((self.moves, moves))

  #def trimGameList(self, generation):
  #  minId = np.min(self.gameIDs)
  #  maxID = np.max(self.gameIDs)
  #  self.meanScore    = np.mean(self.gameScores)
    
  #  if self.meanScore == -50: #if mean score is 0, restart
  #    self.gameStates = np.zeros((0,globe.GRID_X,globe.GRID_Y))
  #    self.gameHeads = np.zeros((0,4))
  #    self.gameScores = np.zeros((0,))
  #    self.gameIDs = np.zeros((0,))
  #    self.moves = np.zeros((0,4))
  #  if (maxID - minId) > globe.NUM_TRAINING_GAMES: #if there are lots of good games, take the best
  #    validIdx        = np.nonzero(self.gameScores > self.meanScore)
  #  else:
  #    validIdx        = np.nonzero(self.gameScores > -50)
  #  self.gameIDs    = self.gameIDs[validIdx]
  #  self.gameScores = self.gameScores[validIdx]
  #  self.gameStates = self.gameStates[validIdx]
  #  self.moves      = self.moves[validIdx]
  #  self.gameHeads  = self.gameHeads[validIdx]
  #  minId = np.min(self.gameIDs)
  #  maxID = np.max(self.gameIDs)
  #  self.meanScores.append(self.meanScore)
  #  self.gamesUsed.append(maxID-minId)
  #  self.maxScores.append(np.max(self.gameScores))
  #  print('Mean score in generation {}: {}. {} games used. Best score of {}.'.format(generation, self.meanScore, self.gamesUsed[-1], self.maxScores[-1]))
  #  return

  def trimGameList(self, generation):
    #get rid of no-score games
    validIdx        = np.nonzero(self.gameScores > -50)
    self.gameIDs    = self.gameIDs[validIdx]
    self.gameScores = self.gameScores[validIdx]
    self.gameStates = self.gameStates[validIdx]
    self.moves      = self.moves[validIdx]
    self.gameHeads  = self.gameHeads[validIdx]

    uni, indices = np.unique(self.gameIDs, return_index=True)
    numberOfGames = len(uni)
    
    meanScore = np.mean(self.gameScores[indices])
    if meanScore > self.meanScore:
      self.meanScore = meanScore

    #trim based on selected id
    #if numberOfGames > globe.NUM_TRAINING_GAMES: #if there are lots of good games, take the best
    validIdx        = np.nonzero(self.gameScores > self.meanScore)
    self.gameIDs    = self.gameIDs[validIdx]
    self.gameScores = self.gameScores[validIdx]
    self.gameStates = self.gameStates[validIdx]
    self.moves      = self.moves[validIdx]
    self.gameHeads  = self.gameHeads[validIdx]

    #update statistics
    numberOfGames = len(np.unique(self.gameIDs))
    self.meanScores.append(meanScore)
    self.gamesUsed.append(numberOfGames)
    self.maxScores.append(np.max(self.gameScores))
    print('Mean score in generation {}: {}. {} games used. Best score of {}.'.format(generation, meanScore, self.gamesUsed[-1], self.maxScores[-1]))
    return

  def sizeInfo(self):
    print('self',       globe.getsize(self))
    print('gameHeads',  globe.getsize(self.gameHeads))
    print('gameids',    globe.getsize(self.gameIDs))
    print('gameScores', globe.getsize(self.gameScores))
    print('gameStates', globe.getsize(self.gameStates))
    print('moves',      globe.getsize(self.moves))