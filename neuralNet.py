import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from globalVar import Globe as globe
from gameState import GameState

class NeuralNetwork:
  def __init__(self):
    self.checkPointID = 0
    self.generationID = 0

    block_input = keras.Input( shape = ( globe.GRID_X, globe.GRID_Y, 1 ), name = 'input_game_state')
    block0 = NeuralNetwork.convBlock(0, block_input)
    block1 = NeuralNetwork.residualBlock(1, block0)
    block2 = NeuralNetwork.residualBlock(2, block1)
    pblock = NeuralNetwork.policyHead(block2)

    self.model = keras.Model(inputs=block_input, outputs=pblock)
    self.compile()

  def evaluate(self, state):
    return self.model.predict(state.reshape(1,state.shape[0],state.shape[1],1))

  def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10, verbose= 0):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return keras.callbacks.LearningRateScheduler(schedule, verbose)

  def compile(self):
    lr_sched = NeuralNetwork.step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)
    self.model.compile(loss={'policy':keras.losses.CategoricalCrossentropy(from_logits=True)},
                  optimizer=keras.optimizers.SGD(momentum=globe.MOMENTUM), 
                  callbacks=[lr_sched])

  def train(self, inputs, predictions, generation):
    history = self.model.fit({'input_game_state':inputs}, {'policy':predictions},
                               batch_size=globe.BATCH_SIZE,
                               epochs=globe.EPOCHS,
                               validation_split=0.15,
                               verbose=2)
    self.save(generation)

  # conv BLOCK
  # convolution 64 filters, 3x3 patch, stride 1
  # batch norm
  # relu
  def convBlock(blockid, block_input):
    l1 = layers.Conv2D( 16, 3, padding = 'same', use_bias = False,        name = 'l1_block_{}'.format( blockid ) )( block_input ) #filters, patch
    l2 = layers.BatchNormalization( axis = -1, momentum = globe.MOMENTUM,  name = 'l2_block_{}'.format( blockid ) )( l1 )
    l3 = layers.Activation( 'sigmoid',                                       name = 'l3_block_{}'.format( blockid ) )( l2 )
    
    return l3

  # RESIDUAL BLOCK
  # convolution 64 filters, 3x3 patch, stride 1
  # batch norm
  # skip connection that adds block input
  # relu

  # convolution 64 filters, 3x3 patch, stride 1
  # batch norm
  # skip connection that adds block input
  # relu
  def residualBlock(blockid, input):
    l1 = layers.Conv2D( 16, 3, padding = 'same', use_bias = False,       name = 'l4_block_{}'.format( blockid ) )( input ) #filters, patch, stride
    l2 = layers.BatchNormalization( axis = -1, momentum = globe.MOMENTUM, name = 'l5_block_{}'.format( blockid ) )( l1 )
    l3 = layers.Activation( 'sigmoid',                                      name='l7_block_{}'.format( blockid ) )( l2 )

    l4 = layers.Conv2D( 16, 3, padding = 'same', use_bias = False,       name = 'l8_block_{}'.format( blockid ) )( l3 ) #filters, patch, stride
    l5 = layers.BatchNormalization( axis = -1, momentum = globe.MOMENTUM, name = 'l9_block_{}'.format( blockid ) )( l4 )
    l6 = layers.add( [ l5, input ],                                      name = 'l10_block_{}'.format( blockid ))
    l7 = layers.Activation( 'sigmoid',                                      name = 'l11_block_{}'.format( blockid ) )( l6 )
    
    return l7

  # policy HEAD
  # convolution 2 filter, 1x1 patch, stride 1
  # batch norm
  # relu
  # fully connected to output
  # relu
  def policyHead(input):
    l1 = layers.Conv2D( 2, 1, padding = 'same', use_bias = False,        name = 'policyhead_conv' )( input )
    l2 = layers.BatchNormalization( axis = -1, momentum = globe.MOMENTUM, name = 'policyhead_batch_norm' )( l1 )
    l3 = layers.Activation( 'sigmoid',                                      name = 'policyhead_activation' )( l2 )
    l4 = layers.GlobalAveragePooling2D(                                  name = 'policyhead_pool')(l3)
    l5 = layers.Dense( 4,  activation = 'sigmoid',                          name = 'policy' )( l4 )
    return l5

  def dispModel(self):
    print( self.model.summary() )
    print( self.model.layers)
    print(self.model.metrics_names)
    #keras.utils.plot_model( self.model, show_shapes = True )

  def save(self,generation):
    self.model.save(r'C:\Users\Chris Minar\Documents\Python\Snake\saves\generation_{}.ckpt'.format(generation))

  def load(self,generation):
    self.model = keras.models.load_model('saves/generation_{}.ckpt'.format(generation))
