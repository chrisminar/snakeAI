"""Handle the neural network."""
import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import layers
from numpy import typing as npt
from tensorflow import keras

from training.helper import (BATCH_SIZE, EPOCH_DELTA, EPOCHS, GRID_X, GRID_Y,
                             LEARNING_RATE, MOMENTUM, SAVE_INTERVAL,
                             VALIDATION_SPLIT, GridEnum)


class NeuralNetwork:
    """Neural network to run snake."""

    def __init__(self, x_size: int = GRID_X, y_size: int = GRID_Y) -> None:
        # weight initializer
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

        # head
        head_input = keras.Input(shape=(4, ), name='head_input')

        # grid side
        grid_input = keras.Input(
            shape=(y_size, x_size, 4), name='grid_input')

        grid_1 = layers.Conv2D(32, 4, 2, padding='same', activation='relu',
                               name='grid_1', kernel_initializer=initializer)(grid_input)

        grid_2 = layers.Conv2D(64, 2, 2, padding='same', activation='relu',
                               name='grid_2', kernel_initializer=initializer)(grid_1)

        reshape_layer = layers.Reshape((-1,))(grid_2)

        grid_3 = layers.Dense(
            512, activation='relu', name='grid_3', kernel_initializer=initializer)(reshape_layer)

        grid_4 = layers.Dense(4, name='grid_4',
                              kernel_initializer=initializer)(grid_3)

        output_layer = layers.Multiply(
            name='output_layer')([grid_4, head_input])

        self.model = keras.Model(
            inputs=[grid_input, head_input], outputs=output_layer)
        self.compile()

    def evaluate(self, *, state: npt.NDArray[np.int32], head: npt.NDArray[np.bool8]) -> npt.NDArray[np.float32]:
        """Evaluate inputs on neural network.

        Args:
            state (npt.NDArray[np.int32]): Grid cells.
            head (npt.NDArray[np.bool8]): Array of if it's valid to move in a given direction.

        Returns:
            npt.NDArray[np.float32]: Confidence that each direction is the best choice.
        """
        converted_states = self.pre_process_input(state)
        if converted_states.ndim == 4:  # evaluating more than one
            head_in = head.reshape(*head.shape, 1).astype(np.float32)
        else:
            head_in = head.reshape(1, 4).astype(np.float32)

        return self.model.predict([converted_states.astype(np.float32), head_in], verbose=0)

    def compile(self) -> None:
        """Compile the neural network."""
        self.model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                           optimizer=keras.optimizers.SGD(
                               momentum=MOMENTUM,
                               learning_rate=LEARNING_RATE),
                           metrics=['accuracy'])

    def train(self,
              *,
              states: npt.NDArray[np.int32],
              heads: npt.NDArray[np.bool8],
              predictions: npt.NDArray[np.float32],
              generation: int,
              verbose: int) -> None:
        """Train the weights and biases of the neural network.

        Args:
            grids (npt.NDArray[np.int32]): Pre-processed snake grids.
            heads (npt.NDArray[np.bool8]): Snake head availibliltiy.
            predictions (npt.NDArray[np.float32]): The move that was chosen at each state.
            generation (int): Neural net generation for this training session.
        """
        pre_processed_states = self.pre_process_input(states)
        if not 0 <= verbose <= 2:
            raise ValueError("Verbosity must be 0,1,or 2.")
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=EPOCH_DELTA, verbose=1, patience=10, restore_best_weights=True)
        log_dir = "media/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        self.model.fit({'grid_input': pre_processed_states, 'head_input': heads}, {'output_layer': predictions},
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_split=VALIDATION_SPLIT,
                       verbose=verbose,
                       callbacks=[early_stop_callback, tensorboard_callback])
        self.save(generation)

    def pre_process_input(self, state: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        """Pre process snake grid.

        (n, x, y) -> (n, x, y, 4)
        Where each new layer is a boolean value where a 1 represents a grid type (see GridEnum).

        Args:
            states (npt.NDArray[np.int32]): (n,x,y) of snake game states.

        Returns:
            npt.NDArray[np.int32]: (n,x,y,4) pre processed nn inputs.
        """
        converted_states = np.zeros((*state.shape, 4), dtype=state.dtype)
        converted_states[:, :, :, 0] = state == GridEnum.HEAD.value
        converted_states[:, :, :, 1] = state == GridEnum.FOOD.value
        converted_states[:, :, :, 2] = state == GridEnum.EMPTY.value
        converted_states[:, :, :, 3] = state >= GridEnum.BODY.value
        return converted_states

    def disp_model(self) -> None:
        """Display model for debugging."""
        print(self.model.summary())
        print(self.model.layers)
        print(self.model.metrics_names)

    def save(self, generation: int) -> None:
        """Save neural network to disk.

        Args:
            generation (int): Training generation of this neural network.
        """
        if generation % SAVE_INTERVAL == 0:
            self.model.save(
                f'media/saves/generation_{generation}.ckpt')

    def load(self, path: Path) -> None:
        """Load neural net from filpath.

        Args:
            path (Path): Path to neural network save file.
        """
        self.model = keras.models.load_model(path)
