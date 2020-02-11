#steps (end of page 5)

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
  # neural network
    # tensorflow blocks
    # BLOCK
    # convolution 64 filters, 3x3 patch, stride 1
    # batch norm
    # relu
    # convolution 64 filters, 3x3 patch, stride 1
    # batch norm
    # skip connection that adds block input
    # relu
    # convolution 64 filters, 3x3 patch, stride 1
    # batch norm
    # skip connection that adds block input
    # relu
    #
    # HEAD
    # 
    # convolution 1 filter, 1x1 patch, stride 1
    # batch norm
    # relu
    # fully connceced to hidden layer size 64
    # relu
    # fully connected linear layer to scaler
    # tanh outputing a scalar in range -1 1
  # using the last x games, train the network
    #how do positive and negative examples work?
    #outputs of neural network?
    #inputs to neural network?
  #neural network (p,v) = f_theta(s) is adjusted to minimse the error between the predicted value v and the self play winner z, and to maximise the similarities between p and pi.
  #loss function l=(z-v)^2-pi^T log(p)+c||theta||^2
  #c = constant = 1e-4
  #stocastic gradient descent with momentum and learning rate annealing
  #momentum = 0.9

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
  # input to self play
    #best theta
  # output of self play
    #25000 games
    #for the first 30 moves the temperature is set to 1, afterwards it is set to 0

# generic architecture
  # parallelism handeled here
  # data storage
    # check point performance
    # best check point
    # most recent x games

# do network pruning
# automated unit testing
# track/understand what the neural network is doing
# gui
# pause/play 
# ORDER

# domain knowledge -- what should the ai know?
  # mcts should attempt to not go into body or wall

# WORK BLOCKS

# learning
  # how to do automatic unit testing with python/git/VS
  # mcts

# YOU ARE HERE.
# WRITE SUDO CODE
# WRITE SKELETAL ARCHITECHTURE
#neural net
  # 2 blocks 64 filters each
  # 1 head policy
  # 1 head value
#self play game
  # game state for each turn
  # end score
  # iteration of a_theta_i that made the game
#self play list
  #list of self play games
#mcts
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



# data inputs for each step
#neural network -- training
  # last 20000 games of self play
#mcts evaluator
  # best neural network
  # current neural network
#self play
  # best neural network

# data outputs for each step
#neural network -- training
  # newest neural network
# mcts evaluator
  # 400 game outputs
  # mean score over 400 games
# self play
  # 2000 game outputs

#bonus stuff to do
  #handle the incorrect data type being passed into dataframes
  #how to manage git from visual studio

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import typing
from typing import List,Tuple

#globals
GRID_X = 10                # x grid size of snake game
GRID_Y = 10                # y grid size of snake game
NUM_EVALUATION_GAMES = 400 # number of evaluation games to play
NUM_SELF_PLAY_GAMES = 400  # number of self play games to play
NUM_TRAINING_GAMES = 20000 # number of self play games to train on

class dataTrack:
  def __init__(self):
    self.self_play_detail_column_names = ['time', 'score',      'generation', 'game_id']
    self.self_play_broad_column_names  = ['time', 'mean score']
    self.evaluator_detail_column_names = ['time', 'score',      'generation', 'game_id', 'game_length', 'mcts_tree_total_time', 'mcts_nn_total_time', 'mcts_tree_mean_time', 'mcts_nn_mean_time']
    self.evaluator_broad_column_names  = ['time', 'mean_score']
    self.training_column_names         = ['time', 'num_minibatch', 'mean_time_per_minibatch']
    self.self_play_detail = pd.DataFrame(columns=self.self_play_detail_column_names) #index is #
    self.self_play_broad  = pd.DataFrame(columns=self.self_play_broad_column_names) #index is generation #
    self.evaluator_detail = pd.DataFrame(columns=self.evaluator_detail_column_names) #index is #
    self.evaluator_broad  = pd.DataFrame(columns=self.evaluator_broad_column_names) #index is #
    self.training         = pd.DataFrame(columns=self.training_column_names) #index is generation #

  def appendSelfPlayDetail(self, time:float, score:int, generation:int, game_id:int):
    current_index = len(self.self_play_detail.index.values)
    self.self_play_detail.loc[current_index] = [time, score, generation, game_id]

  def appendSelfPlayBroad(self, time:float, mean_score:int):
    current_index = len(self.self_play_broad.index.values)
    self.self_play_broad.loc[current_index] = [time, mean_score]

  def appendEvaluatorDetail(self, time:float, score:int, generation:int, game_id:int, game_length:int, mcts_tree_total_time:float, mcts_nn_total_time:float, mcts_tree_mean_time:float, mcts_nn_mean_time:float):
    current_index = len(self.evaluator_detail.index.values)
    self.evaluator_detail.loc[current_index] = [time, score, generation, game_id, game_length, mcts_tree_total_time, mcts_nn_total_time, mcts_tree_mean_time, mcts_nn_mean_time]

  def appendEvaluatorBroad(self, time:float, score:int):
    current_index = len(self.evaluator_broad.index.values)
    self.evaluator_broad.loc[current_index] = [time, score]

  def appendTraining(self, time:float, num_minibatch:int, mean_time_per_minibatch:float):
    current_index = len(self.training.index.values)
    self.training.loc[current_index] = [time, num_minibatch, mean_time_per_minibatch]

#mcts evaluator
#network trainer
#self play

class game_state:
  #game state
  # array gridx x gridy int[][]
  # food 2
  # empty 1
  # body -1
  # head -2
  def __init__(self):
    global GRID_X
    global GRID_Y
    self.x = GRID_X
    self.y = GRID_Y
    self.grid = np.zeros((self.x,self.y))

class neural_network:
  #neural network
    # tensorflow blocks
    # BLOCK
    # convolution 64 filters, 3x3 patch, stride 1
    # batch norm
    # relu
    # convolution 64 filters, 3x3 patch, stride 1
    # batch norm
    # skip connection that adds block input
    # relu
    # convolution 64 filters, 3x3 patch, stride 1
    # batch norm
    # skip connection that adds block input
    # relu
    #
    # HEAD
    # 
    # convolution 1 filter, 1x1 patch, stride 1
    # batch norm
    # relu
    # fully connceced to hidden layer size 64
    # relu
    # fully connected linear layer to scaler
    # tanh outputing a scalar in range -1 1
  def __init__(self):
    self.out = nn_out()

class mcts:
  def __init__(self, state:game_state, neural_net:neural_network):
    self.s = state
    self.f_theta = neural_net
    self.root = mcts_node(self.s)

  def evaluate(self):
    f_theta.evaluate(self.s)
    return self.f_theta.out.move

class mcts_node:
  #mcts - node
  # game state
  # P (s,a) float[]
  # vist count N (s,a) int
  # action value Q(s,a) float
  # children mctsNode[]
  # parent mctsNode
  def __init__(self, state, parent = None):
    self.parent = parent
    self.s = state
    self.P = [0.0]*4
    self.N = 0
    self.Q = 0.0
    self.children = []

class nn_out:
  #new move
    #direction enumerator

  #vector of move probabilities -p
    # 4 array with probabilities for up,right,down,left

  #vector predicted value -v
    # 4 array with predicted score for up,right,down,left
  def __init__(self):
    self.move = '' #direction the neural net thinks the snake should move
    self.P = ['']*4 #move probabilities for each of the 4 directions
    self.V = 0 #predicted score of the game