# todo list
# test for timeout
# change trim function so we don't have the weird zigzaging, ie try to have a constant number of samples once saturated
# try to progressively update nn, not reinit every time?
# possible failure of trim game states after too many runs (numer of samples constantly increases)
  #try to print out gameid length every game.
# try init snake in middle (not 0,0)
# possible stagnation around 250 score

# try training on more epochs? -- seems to stop benefitting around 2

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
    return

  def train(self):
    generation = 0
    while 1:
      self.selfPlay(self.nn, generation)
      self.networkTrainer(generation)
      print(generation, globe.getsize(self.nn))
      generation += 1
    return

  #operates on:
    #neural network
  #outputs:
    #2000 game outputs
  def selfPlay(self, nn:NeuralNetwork, generation:int):
    spc = SelfPlay(nn)
    states, heads, scores, ids, moves = spc.playGames(generation, self.gameID)
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

  def trimGameList(self, generation):
    minId = np.min(self.gameIDs)
    maxID = np.max(self.gameIDs)
    self.meanScore    = np.mean(self.gameScores)
    print('Mean score in generation {}: {}'.format(generation, self.meanScore))
    if self.meanScore == -50: #if mean score is 0, restart
      self.gameStates = np.zeros((0,globe.GRID_X,globe.GRID_Y))
      self.gameHeads = np.zeros((0,4))
      self.gameScores = np.zeros((0,))
      self.gameIDs = np.zeros((0,))
      self.moves = np.zeros((0,4))
    if (maxID - minId) > globe.NUM_TRAINING_GAMES: #if there are lots of good games, take the best
      validIdx        = np.nonzero(self.gameScores > self.meanScore)
    else:
      validIdx        = np.nonzero(self.gameScores > -50)
    self.gameIDs    = self.gameIDs[validIdx]
    self.gameScores = self.gameScores[validIdx]
    self.gameStates = self.gameStates[validIdx]
    self.moves      = self.moves[validIdx]
    self.gameHeads  = self.gameHeads[validIdx]
    return

  def sizeInfo(self):
    print('self',       globe.getsize(self))
    print('gameHeads',  globe.getsize(self.gameHeads))
    print('gameids',    globe.getsize(self.gameIDs))
    print('gameScores', globe.getsize(self.gameScores))
    print('gameStates', globe.getsize(self.gameStates))
    print('moves',      globe.getsize(self.moves))