
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

    #inputs_  = keras.Input(shape=(gridX,gridY))
    #outputs_ = self.HeadBlock()
    #self.model    = keras.Model(inputs=inputs_, outputs=outputs)
    temp = self.block(0)
    self.dispModel(temp)

  def block(self, blockid):
    global MOMENTUM
    gridX = 8
    gridY = 8
    block_input = keras.Input(shape = (gridX,gridY), name='block{}'.format(blockid))
    l1 = layers.Conv2D(64, 3, strides=(1,1), use_bias=False)(block_input) #filters, patch, stride
    l2 = layers.BatchNormalization(axis=1, momentum = MOMENTUM)(l1)
    l3 = layers.Activation('relu')(l2)

    l4 = layers.Conv2D(64, 3, strides=(1,1), use_bias=False)(l3) #filters, patch, stride
    l5 = layers.BatchNormalization(axis=1, momentum = MOMENTUM)(l4)
    l6 = layers.merge([l5,block_input],mode = 'sum')
    l7 = layers.Activation('relu')(l6)

    l8 = layers.Conv2D(64, 3, strides=(1,1), use_bias=False)(l7) #filters, patch, stride
    l9 = layers.BatchNormalization(axis=1, momentum = MOMENTUM)(l8)
    l10 = layers.merge([l9,block_input],mode = 'sum')
    l11 = layers.Activation('relu')(l10)
    
    block_model = keras.Model(block_input, l11)
    return block_model

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
