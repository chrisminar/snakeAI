#steps (end of page 5)

#glossary
#p: vector of move probabilities
#v: predictied value
#s: board state
#pi: mcts probabilities of playing each move
#f_theta: neural network
#z game winner
#a_ 25000
#b_ 1600
#c_ 500000

# sweet things to do
  # features
    # do network pruning
    # automated unit testing?
    # track/understand what the neural network is doing
    # gui
    # pause/play 
  # design processes
    # make all the data handled with pandas
    # how can parallelism be abused to make this faster?
    # develop with unit testing as I go
  # evaluation
    # sweep the number of blocks vs performance
    # do something with computational power vs performance

# domain knowledge -- what should the ai know?
  # mcts should attempt to not go into body or wall

# WORK BLOCKS

# learning
  # how to pandas
  # how to do automatic unit testing with python/git/VS
  # mcts

#YOU ARE HERE. FIGURE OUT MORE SPECIFIC SIZES FOR EVERYTHING AND FLUSH OUT DATATYPES
#THEN START WRITING SUDO CODE
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

# data tracking
# neural network -- training
  # time -- minibatch
  # time -- generation
# neural network -- feedforward
  
# mcts evaluator
  # time -- game
  # time -- generation
# mcts
  # time -- tree generation
  # time -- single ffnn
# self play
  # time -- generation

# neural network trainer -- update neural network parameters theta_i -using recent self play data
  # neural network
    # tensorflow blocks
    # BLOCK
    # convolution 256 filters, 3x3 patch, stride 1
    # batch norm
    # relu
    # convolution 256 filters, 3x3 patch, stride 1
    # batch norm
    # skip connection that adds block input
    # relu
    # convolution 256 filters, 3x3 patch, stride 1
    # batch norm
    # skip connection that adds block input
    # relu
    #
    # HEAD
    # 
    # convolution 1 filter, 1x1 patch, stride 1
    # batch norm
    # relu
    # fully connceced to hidden layer size 256
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