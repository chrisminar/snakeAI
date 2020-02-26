#glossary
#p: vector of move probabilities (likelyhood of choosing a given move)
#v: predictied victor of the game
#s: board state
#pi: mcts probabilities of playing each move
#f_theta: neural network
#z game winner
#a_ 25000
#b_ 1600
#c_ 500000

# evaluation
  # sweep the number of blocks vs performance
  # do something with computational power vs performance

# neural network trainer -- update neural network parameters theta_i -using recent self play data
  # using the last x games, train the network
    #how do positive and negative examples work?
    #outputs of neural network?
    #inputs to neural network?
  #neural network (p,v) = f_theta(s) is adjusted to minimse the error between the predicted value v and the self play winner z, and to maximise the similarities between p and pi.
  #loss function l=(z-v)^2-pi^T log(p)+c||theta||^2
  #c = constant = 1e-4

# mcts evaluator -- alpha go players a_theta_i are evaluated and best performing generates new self play data
  # game terminates when staleness threshold is passed, or when score exceeds a certain length
  # MCTS -- monte carlo tree search
    # inputs to mcts?
    # outputs to mcts?
    # data storage/tracking?
  # inputs to mcts evaluator
  # outputs to mcts evaluator
  # checkpoint creator -- save best theta
  # checkpoint loader -- load best theta
  # checkpoint every 1000 steps
  # evaluation consists of 400 games
  #if checkpoint_t scores 1.2x> checkpoint_(i_best), it is the new baseline and used to generate data
  #mcts
    # store:
      # node
        # state
        # Prior probability P (s,a)
        # vist count N (s,a)
        # action value Q(s,a)
        # children
        # parent
    # selection -- starts from root and selects children nodes until a leaf is found.
      #maximise confidence bound to traverse the tree
      #Q(s,a) + U(s,a)
      #U(s,a) = P(s,a)/(1+N(s,a))
    # expansion -- make a new node from leaf using output from neural net f_theta
    # simulation -- play out a game somehow and determine a winner
      # P(s'), V(s') = f_theta(s') --- probability and prediction from new state with neural net f_theta
    # backpropagation -- update the tree using the winner of simulation
      #each node (s,a) is traversed to update:
        #N(s,a) += 1
        #Q(s,a) = 1/N(s,a) Sum_s'|s,a->s' V(s')
          #s,a->s' indicates a simulation reached s' after taking move a from position s (sum the number of times this node was chosen)
# self play
    #for the first 30 moves the temperature is set to 1, afterwards it is set to 0

# do network pruning
# track/understand what the neural network is doing
# gui
# pause/play 

# domain knowledge -- what should the ai know?
  # mcts should attempt to not go into body or wall

# WORK BLOCKS
# ON DECK
  # UNIT TEST
  # add predictions to tracked data fron snakeRL
  # MCTS
    # make autoplay loop in snakerl
  # EVALUATOR

# self play game
  # game state for each turn
  # end score
  # iteration of a_theta_i that made the game
# self play list
  # list of self play games
# mcts
  # state
  # Prior probability P (s,a)
  # vist count N (s,a)
  # action value Q(s,a)
  # children
  # parent

# data tracking -- put this in pandas datastructs
# generation
  # self play
    # 2000 games
      # time
      # score
    # generation time
    # mean score
  # mcts evaluator
    # 400 games
      # time of game
      # score of game
      # number of moves taken
      # total time for mcts tree generation
      # total time for ffnn
      # mean time for mcts tree generation
      # mean time for ffnn
    # mean score
    # generation time
  # training -- ammount of training data per generation will not be consistent due to different games lengths so don't track mini-batch-wise statistics
    # 20000 training games
    # total training time
    # mean training time per mini batch

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import typing
from typing import List,Tuple
from dataTrack import dataTrack
from mcts import mcts
from mcts import mcts_node
from neuralNet import neural_network
from neuralNet import nn_out
from game_state import game_state
from globalVar import globe

from selfPlayClass import selfPlayClass
from trainer import trainerClass

class snakeRL():
  def __init__(self):
    self.tracker = dataTrack()
    self.gameStates = np.zeros((0,globe.GRID_X,globe.GRID_Y))
    self.gameScores = np.zeros((0,1))
    self.gameIDs = np.zeros((0,1))
    self.nnList = []
    self.gameID = 0
    pass


  def train(self):
    generation = 0
    while 1:
      self.selfPlay(generation)
      self.networkTrainer(generation)
      self.mcts_evaluator()
      generation += 1

  #operates on:
    #best neural network
  #outputs:
    #2000 game outputs
  def selfPlay(self, nn:neural_network, generation:int):
    spc = selfPlayClass(self.tracker, nn)
    states, scores, ids = spc.playGames(generation, self.gameID)
    self.addGamesToList(states, scores, ids)
    self.trimGameList()

  #operates on:
    # last 20000 games of self play
  #outputs:
    #new neural network
  def networkTrainer(self, generation:int):
    trn = trainerClass(self.tracker)
    trn.train(generation, self.nnList[-1], self.gameStates, self.gameScores)
    pass

   #operates on: 
    #best neural network
    #current neural network
  #outputs:
    #400 games (with gamestate)
    #mean score of 400 games
  def mcts_evaluator(self):
    pass

  def addGamesToList(self, states, scores, ids):
    self.gameStates = np.concatenate(self.gameStates, states)
    self.gameScores = np.concatenate(self.gameScores, scores)
    self.gameIDs = np.concatenate(self.gameIDs, ids)

  def trimGameList(self):
    minId = np.min(self.gameIDs)
    maxID = np.max(self.gameIDs)
    if maxID - minId > globe.NUM_TRAINING_GAMES:
      validIdx = np.argwhere(self.gameIDs > maxID - globe.NUM_TRAINING_GAMES)
      self.gameIDs = self.gameIDs[validIdx]
      self.gameScores = self.gameScores[validIdx]
      self.gameStates = self.gameStates[validIdx,:,:]
    return

#######################
## network functions ##
#######################
#load checkpoint
#train network
#save checkpoint
#add nn to nn list
#add statistics

####################
## mcts evaluator ##
####################
#add statistics