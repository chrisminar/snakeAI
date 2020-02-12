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
    self.checkPointID = 0
    self.generationID = 0
    self.out = nn_out()

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
