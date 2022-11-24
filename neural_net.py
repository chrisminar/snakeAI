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
    """Neural network to run snake."""

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

        layer_1 = layers.Conv2D(16, 3, padding='same', activation='relu',
                                name='l1', kernel_initializer=initializer)(block_input)

        layer_2 = layers.Conv2D(16, 3, padding='same', activation='relu',
                                name='l2', kernel_initializer=initializer)(layer_1)

        layer_3 = layers.Conv2D(4, 1, padding='same', activation='relu',
                                name='l4', kernel_initializer=initializer)(layer_2)

        layer_4 = layers.GlobalAveragePooling2D(name='pool')(layer_3)

        layer_5 = layers.BatchNormalization(name='norm')(layer_4)

        # combine
        layer_6 = layers.add([layer_5, l4_head], name='add')

        layer_y = layers.Dense(4,  activation='relu', name='last_fully_connected',
                               kernel_initializer=initializer)(layer_6)

        layer_8 = layers.Softmax(name='policy')(layer_y)

        layer_9 = layers.Multiply(name='mult')([layer_8, head_input])

        self.model = keras.Model(
            inputs=[block_input, head_input], outputs=layer_9)
        self.compile()

    def evaluate(self, state: npt.NDArray[np.int32], head: npt.NDArray[np.bool8]) -> npt.NDArray[np.float32]:
        """Evaluate inputs on neural network.

        Args:
            state (npt.NDArray[np.int32]): Grid cells.
            head (npt.NDArray[np.bool8]): Array of if it's valid to move in a given direction.

        Returns:
            npt.NDArray[np.float32]: Confidence that each direction is the best choice.
        """
        grid_in = state.reshape(
            1, state.shape[0], state.shape[1], 1).astype(np.float32)
        head_in = head.reshape(1, 4).astype(np.float32)
        return self.model([grid_in, head_in], training=False)

    def compile(self) -> None:
        """Compile the neural network."""
        self.model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                           optimizer=keras.optimizers.SGD(
                               momentum=MOMENTUM),
                           metrics=['accuracy', 'accuracy'])

    def train(self,
              grids: npt.NDArray[np.int32],
              heads: npt.NDArray[np.bool8],
              predictions: npt.NDArray[np.float32],
              generation: int) -> None:
        """Train the weights and biases of the neural network.

        Args:
            grids (npt.NDArray[np.int32]): Pre-processed snake grids.
            heads (npt.NDArray[np.bool8]): Snake head availibliltiy.
            predictions (npt.NDArray[np.float32]): The move that was chosen at each state.
            generation (int): Neural net generation for this training session.
        """
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=EPOCH_DELTA, verbose=1)
        self.model.fit({'input_game_state': grids, 'input_head': heads}, {'mult': predictions},
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_split=VALIDATION_SPLIT,
                       verbose=0,
                       callbacks=[callback])
        self.save(generation)

    def disp_model(self) -> None:
        """Display model for debugging."""
        print(self.model.summary())
        print(self.model.layers)
        print(self.model.metrics_names)
        # keras.utils.plot_model( self.model, show_shapes = True )

    # TODO this path shouldn't be hardcoded
    def save(self, generation: int) -> None:
        """Save neural network to disk.

        Args:
            generation (int): Training generation of this neural network.
        """
        self.model.save(
            fr'C:\Users\Chris Minar\Documents\Python\Snake\saves\generation_{generation}.ckpt')

    def load(self, path: Path) -> None:
        """Load neural net from filpath.

        Args:
            path (Path): Path to neural network save file.
        """
        self.model = keras.models.load_model(path)
