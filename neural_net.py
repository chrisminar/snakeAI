"""Handle the neural network."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import layers
from numpy import typing as npt
from tensorflow import keras

from helper import (BATCH_SIZE, EPOCH_DELTA, EPOCHS, GRID_X, GRID_Y, MOMENTUM,
                    VALIDATION_SPLIT)


class NeuralNetwork:
    def __init__(self) -> None:
        # weight initializer
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

        # head side
        head_input = keras.Input(shape=(4, ), name='input_head')

        l2_head = layers.Dense(
            32, activation='relu', name='l2_head', kernel_initializer=initializer)(head_input)

        l3_head = layers.Dense(
            4, activation='relu', name='l3_head', kernel_initializer=initializer)(l2_head)

        l4_head = layers.BatchNormalization(name='head_norm')(l3_head)

        # grid side
        block_input = keras.Input(
            shape=(GRID_X, GRID_Y, 1), name='input_game_state')

        l1 = layers.Conv2D(16, 3, padding='same', activation='relu',
                           name='l1', kernel_initializer=initializer)(block_input)

        l2 = layers.Conv2D(16, 3, padding='same', activation='relu',
                           name='l2', kernel_initializer=initializer)(l1)

        l3 = layers.Conv2D(4, 1, padding='same', activation='relu',
                           name='l4', kernel_initializer=initializer)(l2)

        l4 = layers.GlobalAveragePooling2D(name='pool')(l3)

        l5 = layers.BatchNormalization(name='norm')(l4)

        # combine
        l5 = layers.add([l5, l4_head], name='add')

        l6 = layers.Dense(4,  activation='relu', name='last_fully_connected',
                          kernel_initializer=initializer)(l5)

        l7 = layers.Softmax(name='policy')(l6)

        l8 = layers.Multiply(name='mult')([l7, head_input])

        self.model = keras.Model(inputs=[block_input, head_input], outputs=l8)
        self.compile()

    def evaluate(self, state: npt.NDArray[np.int32], head: npt.NDArray[np.int32]) -> None:
        grid_in = state.reshape(
            1, state.shape[0], state.shape[1], 1).astype(np.float32)
        head_in = head.reshape(1, 4).astype(np.float32)
        return self.model([grid_in, head_in], training=False)

    def compile(self) -> None:
        self.model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                           optimizer=keras.optimizers.SGD(
                               momentum=MOMENTUM),
                           metrics=['accuracy', 'accuracy'])

    def train(self, inputs: npt.NDArray[np.int32],
              heads: npt.NDArray[np.int32],
              predictions: npt.NDArray[np.int32],
              generation: int,) -> None:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=EPOCH_DELTA, verbose=1)
        history = self.model.fit({'input_game_state': inputs, 'input_head': heads}, {'mult': predictions},
                                 batch_size=BATCH_SIZE,
                                 epochs=EPOCHS,
                                 validation_split=VALIDATION_SPLIT,
                                 verbose=0,
                                 callbacks=[callback])
        # debug
        # tempM = []
        # x = []
        # for layer in self.model.layers:
        #  tempM.append( keras.Model(inputs=self.model.input, outputs=layer.output) )
        # for i in range(len(tempM)):
        #  x.append( tempM[i].predict( [inputs[0].reshape(1,inputs[0].shape[0],inputs[0].shape[1],1).astype(np.float32), heads[0].reshape(1,4).astype(np.float32)]) )
        self.save(generation)

    def disp_model(self) -> None:
        print(self.model.summary())
        print(self.model.layers)
        print(self.model.metrics_names)
        # keras.utils.plot_model( self.model, show_shapes = True )

    def save(self, generation: int) -> None:
        self.model.save(
            r'C:\Users\Chris Minar\Documents\Python\Snake\saves\generation_{}.ckpt'.format(generation))

    def load(self, path: Path) -> None:
        self.model = keras.models.load_model(path)
