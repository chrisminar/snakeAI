#steps (end of page 5)
#play a_ games using b_ monte-carlo rollout per move (explained at the bottom of page 25)
#
#optimize loss function _ on last c_ games
#
#
# game terminates when staleness threshold is passed, or when score exceeds a certain length
#
#TRAINING
#
#neural network (p,v) = f_theta(s) is adjusted to minimse the error between the predicted value v and the self play winner z, and to maximise the similarities between p and pi.
#loss function l=(z-v)^2-pi^T log(p)+c||theta||^2
#c = constant = 1e-4
#stocastic gradient descent with momentum and learning rate annealing
#momentum = 0.9
#
#Evaluation
#
#checkpoint every 1000 steps
#evaluation consists of 400 games
#if checkpoint_t scores 1.2x> checkpoint_(t-1), it is the new baseline and used to generate data
#
#SELF PLAY
#25000 games
#for the first 30 moves the temperature is set to 1, afterwards it is set to 0
#
#ARCHITECTURE
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
#
#
#
#
#
#
#
#
# 
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