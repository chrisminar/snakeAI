
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers


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

    #inputs_  = keras.Input(shape=(gridX,gridY))
    #outputs_ = self.HeadBlock()
    #self.model    = keras.Model(inputs=inputs_, outputs=outputs)
    gridX = 8
    gridY = 8
    block_input = keras.Input(shape = (gridX,gridY,1), name='input_game_state')
    block0 = self.block(0, block_input)
    block1 = self.block(1, block0)

    self.model = block_model = keras.Model(block_input, block1)

  def block(self, blockid, input):
    MOMENTUM = 0.9
    l1 = layers.Conv2D(64, 3, padding = 'same', use_bias=False, name='l1_block_{}'.format(blockid))(input) #filters, patch
    l2 = layers.BatchNormalization(axis=1, momentum = MOMENTUM, name='l2_block_{}'.format(blockid))(l1)
    l3 = layers.Activation('relu',                              name='l3_block_{}'.format(blockid))(l2)

    l4 = layers.Conv2D(64, 3, padding = 'same', use_bias=False, name='l4_block_{}'.format(blockid))(l3) #filters, patch, stride
    l5 = layers.BatchNormalization(axis=1, momentum = MOMENTUM, name='l5_block_{}'.format(blockid))(l4)
    l6 = layers.add([l5,input],                                 name='l6_block_{}'.format(blockid))
    l7 = layers.Activation('relu',                              name='l7_block_{}'.format(blockid))(l6)

    l8 = layers.Conv2D(64, 3, padding = 'same', use_bias=False, name='l8_block_{}'.format(blockid))(l7) #filters, patch, stride
    l9 = layers.BatchNormalization(axis=1, momentum = MOMENTUM, name='l9_block_{}'.format(blockid))(l8)
    l10 = layers.add([l9,input],                                name='l10_block_{}'.format(blockid))
    l11 = layers.Activation('relu',                             name='l11_block_{}'.format(blockid))(l10)
    
    return l11

  def head(self):
    pass #YOU ARE HERE

  def dispModel(self):
    print(self.model.summary())
    keras.utils.plot_model(self.model, show_shapes=True)


    


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
