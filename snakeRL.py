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
  # how to pandas
  # how to do automatic unit testing with python/git/VS
  # mcts

# YOU ARE HERE. FIGURE OUT MORE SPECIFIC SIZES FOR MCST
# FIGURE OUT THE DATA STRUCTS TO HOLD ALL OF THIS INFORMATION
  # dataframes -- done
  # working lists/arrays -- todo
# WHAT IS THE PURPOSE OF 2 HEADS?
# what is the purpose of having distinct evaluations and self play?
# WRITE SUDO CODE
# WRITE SKELETAL ARCHITECHTURE
#game state
  # array gridX x gridY
    # food 2
    # empty 1
    # body -1
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

#new move
  #direction enumerator
#vector of move probabilities -p
  # 4 array with probabilities for up,right,down,left
#vector predicted value -v
  # 4 array with predicted value for up,right,down,left

# data inputs for each step
#neural network -- training
  # last 20000 games of self play
# neural network -- feedforward
  # game state
#mcts evaluator
  # best neural network
  # current neural network
#mcts
  # game state
  # neural network
#self play
  # best neural network

# data outputs for each step
#neural network -- training
  # newest neural network
# neural network -- feedforward
  # vector of move probabilities -p
  # predicted value -v
# mcts evaluator
  # 400 game outputs
  # mean score over 400 games
# mcts
  # new move
# self play
  # 2000 game outputs

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
  
  # dataframes:
  # 1 self play detail
    # columns: time(float),score(int)
    # index1: generation(list[0-?])
    # index2: games(list[0-1999])
  # 2 self play broad
    # columns: mean score(float), time(float)
    # index: generation(list[0-?])
  # 3 evaluator detailed
    # columns: 
      # time of game(float)
      # score of game(int)
      # #moves(int)
      # total time for mcts tree generation(float)
      # total time for ffnn(float)
      # mean time for mcts tree generation (float)
      # mean time for ffnn (float)
    # index:
      #1: generation(list[0-?])
      #2: games(list[0-399])
  # 4 evaluator broad
    #columns:
      #mean score(float)
      #generation time (float)
    #index generation(list[0-?])
  #5 training
    #columns: 
      # generation time (float)
      # # minibatches(int)
      # mean training time per mini batch(float)
    #index:
      #generation(list[0-?])






import pandas as pd
import numpy as np
from matplotlib import pyplot as plt