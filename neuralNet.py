import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from globalVar import Globe as globe

class NeuralNetwork:
  def __init__(self):
    self.checkPointID = 0
    self.generationID = 0
    self.out = nn_out()

    gridX = 8
    gridY = 8
    block_input = keras.Input( shape = ( globe.GRID_X, globe.GRID_Y, 1 ), name = 'input_game_state')
    block0 = self.convBlock(0, block_input)
    block1 = self.residualBlock(1, block0)
    block2 = self.residualBlock(2, block1)
    pblock = self.policyHead(block2)
    vblock = self.valueHead(block2)

    self.model = block_model = keras.Model(inputs=block_input, outputs=[pblock,vblock])
    self.compile()

  def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return keras.callbacks.LearningRateScheduler(schedule)

  def compile(self):
    lr_sched = neural_network.step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)
    model.compile(loss={'policy':keras.losses.CategoricalCrossEntropy(from_logits=True), 
                        'value':keras.losses.MSE},
                  optimizer=keras.optimizers.SGD(momentum=globe.MOMENTUM), 
                  callbacks=[lr_sched],
                  loss_weights=[1,0.1])

  def train(self, inputs, scores, predictions, generation):
    history = model.fit({'input_game_state':inputs, 'policy':predictions, 'value':scores},
                        batch_size=globe.BATCH_SIZE,
                        epochs=globe.EPOCHS,
                        validation_split=0.15)
    self.save(generation)

  # conv BLOCK
  # convolution 64 filters, 3x3 patch, stride 1
  # batch norm
  # relu
  def convBlock(self, blockid, block_input):
    l1 = layers.Conv2D( 64, 3, padding = 'same', use_bias = False,        name = 'l1_block_{}'.format( blockid ) )( block_input ) #filters, patch
    l2 = layers.BatchNormalization( axis = 1, momentum = globe.MOMENTUM,  name = 'l2_block_{}'.format( blockid ) )( l1 )
    l3 = layers.Activation( 'relu',                                       name = 'l3_block_{}'.format( blockid ) )( l2 )
    
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
  def residualBlock(self, blockid, input):
    l1 = layers.Conv2D( 64, 3, padding = 'same', use_bias = False,       name = 'l4_block_{}'.format( blockid ) )( input ) #filters, patch, stride
    l2 = layers.BatchNormalization( axis = 1, momentum = globe.MOMENTUM, name = 'l5_block_{}'.format( blockid ) )( l1 )
    l3 = layers.Activation( 'relu',                                      name='l7_block_{}'.format( blockid ) )( l2 )

    l4 = layers.Conv2D( 64, 3, padding = 'same', use_bias = False,       name = 'l8_block_{}'.format( blockid ) )( l3 ) #filters, patch, stride
    l5 = layers.BatchNormalization( axis = 1, momentum = globe.MOMENTUM, name = 'l9_block_{}'.format( blockid ) )( l4 )
    l6 = layers.add( [ l5, input ],                                      name = 'l10_block_{}'.format( blockid ))
    l7 = layers.Activation( 'relu',                                      name = 'l11_block_{}'.format( blockid ) )( l6 )
    
    return l7

  # policy HEAD
  # convolution 2 filter, 1x1 patch, stride 1
  # batch norm
  # relu
  # fully connected to output
  # relu
  def policyHead(self, input):
    l1 = layers.Conv2D( 2, 1, padding = 'same', use_bias = False,        name = 'policyhead_conv' )( input )
    l2 = layers.BatchNormalization( axis = 1, momentum = globe.MOMENTUM, name = 'policyhead_batch_norm' )( l1 )
    l3 = layers.Activation( 'relu',                                      name = 'policyhead_activation' )( l2 )
    l4 = layers.GlobalAveragePooling2D(                                  name = 'policyhead_pool')(l3)
    l5 = layers.Dense( 4,  activation = 'relu',                          name = 'policy' )( l4 )
    return l5

  # valueHEAD
  # convolution 2 filter, 1x1 patch, stride 1
  # batch norm
  # relu
  # fully connected to hidden layer size 64
  # relu
  # fully connect to size 1
  # tanh activation
  def valueHead(self, input):
    l1 = layers.Conv2D( 2, 1, padding = 'same', use_bias = False,        name = 'valuehead_conv' )( input )
    l2 = layers.BatchNormalization( axis = 1, momentum = globe.MOMENTUM, name = 'valuehead_batch_norm' )( l1 )
    l3 = layers.Activation( 'relu',                                      name = 'valuehead_activation' )( l2 )
    l4 = layers.GlobalAveragePooling2D(                                  name = 'valuehead_pool')(l3)
    l5 = layers.Dense( 64, activation = 'relu',                          name = 'valuehead_dense')( l4 )
    l6 = layers.Dense( 1, activation = 'relu', kernel_regularizer=regularizers.l2(0.0001), name = 'value' )( l5 )

    return l6

  def dispModel(self):
    print( self.model.summary() )
    keras.utils.plot_model( self.model, show_shapes = True )

  def save(self,generation):
    self.model.save('saves/generation_{}.ckpt'.format(generation))

  def load(self,generation):
    self.model = keras.models.load_model('saves/generation_{}.ckpt'.format(generation))


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
