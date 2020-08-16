import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import layers
from globalVar import Globe as globe
from gameState import GameState

class NeuralNetwork:
  def __init__(self):
    self.checkPointID = 0
    self.generationID = 0

    #weight initializer
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev = 1.0)

    #head side
    head_input = keras.Input( shape = ( 4, ), name = 'input_head')

    l2_head = layers.Dense( 32, activation='relu', name='l2_head', kernel_initializer=initializer)(head_input)

    l3_head = layers.Dense( 4, activation='relu', name='l3_head', kernel_initializer=initializer)(l2_head)

    l4_head = layers.BatchNormalization(name='head_norm')(l3_head)

    #grid side
    block_input = keras.Input( shape = ( globe.GRID_X, globe.GRID_Y, 1 ), name = 'input_game_state')

    l1 = layers.Conv2D( 16, 3, padding = 'same', activation = 'relu', name = 'l1', kernel_initializer=initializer )( block_input )

    l2 = layers.Conv2D( 16, 3, padding = 'same', activation = 'relu', name = 'l2', kernel_initializer=initializer )( l1 )

    l3 = layers.Conv2D( 4, 1, padding = 'same', activation = 'relu', name = 'l4', kernel_initializer=initializer )( l2 )

    l4 = layers.GlobalAveragePooling2D( name = 'pool')( l3 )

    l5 = layers.BatchNormalization(name='norm')(l4)

    #combine
    l5 = layers.add( [l5, l4_head], name='add')

    l6 = layers.Dense( 4,  activation = 'relu', name = 'last_fully_connected', kernel_initializer=initializer )( l5 )

    l7 = layers.Softmax(name = 'policy' ) (l6)
    
    self.model = keras.Model(inputs=[block_input, head_input], outputs=l7)
    self.compile()

  def evaluate(self, state, head):
    return self.model.predict( [state.reshape(1,state.shape[0],state.shape[1],1).astype(np.float32), head.reshape(1,4).astype(np.float32)])

  def compile(self):
    self.model.compile(loss={'policy':keras.losses.CategoricalCrossentropy(from_logits=True)},
                  optimizer=keras.optimizers.SGD(momentum=globe.MOMENTUM))

  def train(self, inputs, heads, predictions, generation):
    history = self.model.fit({'input_game_state':inputs, 'input_head':heads}, {'policy':predictions},
                               batch_size=globe.BATCH_SIZE,
                               epochs=globe.EPOCHS,
                               validation_split=0.15,
                               verbose=2)
    #debug
    #tempM = []
    #x = []
    #for layer in self.model.layers:
    #  tempM.append( keras.Model(inputs=self.model.input, outputs=layer.output) )
    #for i in range(len(tempM)):
    #  x.append( tempM[i].predict( [inputs[0].reshape(1,inputs[0].shape[0],inputs[0].shape[1],1).astype(np.float32), heads[0].reshape(1,4).astype(np.float32)]) )
    self.save(generation)

  def dispModel(self):
    print( self.model.summary() )
    print( self.model.layers)
    print(self.model.metrics_names)
    #keras.utils.plot_model( self.model, show_shapes = True )

  def save(self,generation):
    self.model.save(r'C:\Users\Chris Minar\Documents\Python\Snake\saves\generation_{}.ckpt'.format(generation))

  def load(self,generation):
    self.model = keras.models.load_model('saves/generation_{}.ckpt'.format(generation))
