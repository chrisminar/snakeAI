"""Reinforcement learning."""

import logging
from typing import List, Sequence

import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt

from training.helper import (GRID_X, GRID_Y, NUM_SELF_PLAY_GAMES,
                             NUM_TRAINING_GAMES, SCORE_PER_FOOD, Direction,
                             get_size)
from training.neural_net import NeuralNetwork
from training.play_games import PlayBig, PlayGames
from training.trainer import train

LOGGER = logging.getLogger("terminal")


class TrainRL:
    """Reinforcement learning loop."""

    def __init__(self) -> None:
        self.game_states = np.zeros((0, GRID_Y, GRID_X), dtype=np.int32)
        self.game_heads = np.zeros((0, 4), dtype=np.bool8)
        self.game_scores = np.zeros((0,), dtype=np.int32)
        self.game_ids = np.zeros((0,), dtype=np.int32)
        self.moves = np.zeros((0, 4), dtype=np.float32)
        self.neural_net = NeuralNetwork()
        self.game_id = 0
        self.mean_score = -SCORE_PER_FOOD
        self.mean_scores: List[int] = []
        self.games_used: List[int] = []
        self.max_scores: List[int] = []

    def train(self) -> None:
        """Training loop."""
        generation = 0
        while 1:
            LOGGER.info("")
            LOGGER.info("")
            LOGGER.info("Generation %d", generation)
            #num_games = NUM_TRAINING_GAMES - len(np.unique(self.game_ids))
            # if num_games > NUM_SELF_PLAY_GAMES:  # if we need many more games, play many more games
            #    pass
            # else:  # if we already have a lot of games, use default amount
            num_games = NUM_SELF_PLAY_GAMES
            LOGGER.info("Playing %d games.", num_games)
            self.play_one_generation_of_games(
                self.neural_net, generation=generation, num_games=num_games)
            self.trim_game_list()
            self.neural_net = NeuralNetwork()
            self.neural_net = train(
                generation, self.game_states, self.game_heads, self.moves)
            self.gen_status_plot(generation)
            generation += 1

    def gen_status_plot(self, generation: int) -> None:
        """Plot status.

        Args:
            generation (int): Generation number.
        """
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(self.mean_scores)
        plt.title('Mean Score')

        plt.subplot(3, 1, 2)
        plt.plot(self.max_scores)
        plt.title('Max Score')

        plt.subplot(3, 1, 3)
        plt.plot(self.games_used)
        plt.title('Training Games')

        plt.suptitle(f'Generation: {generation}')

        fig.savefig('media/pictures/mostrecent.png')
        plt.close()

    def gen_histogram(self, *, scores: Sequence[np.int32], generation: int) -> None:
        """Plot histogram of game scores.

        Args:
            unique (npt.NDArray[np.int32]): Game scores
            generation (int): Most recent generation.
        """
        fig = plt.figure()
        plt.title(f'Generation {generation}')
        plt.hist(scores, bins=np.arange(start=0, stop=1700, step=100))
        fig.savefig(f'media/hists/generation_{generation}.png')
        plt.close()

    def play_one_generation_of_games(self, neural_net: NeuralNetwork, *, generation: int, num_games: int = NUM_SELF_PLAY_GAMES) -> None:
        """Play one generation worth of snake games.

        Args:
            neural_net (NeuralNetwork): Neural network to play games with.
            generation (int): Generation number.
            num_games (int, optional): Number of games to play. Defaults to NUM_SELF_PLAY_GAMES.
        """
        #spc = PlayGames(neural_net)
        counter = 0
        while 1:  # play games until a generation has at least one succesful game
            counter += 1
            spc = PlayBig(neural_network=neural_net)
            minimum_score = 0 if self.game_scores.size == 0 else self.game_scores.min()
            states, heads, scores, ids, moves = spc.play_games(
                start_id=self.game_id, num_games=num_games, minimum_score=minimum_score, exploratory=True)
            if ids.size > 0:
                break
        LOGGER.debug(
            "Took %d generations to create a game above the minimum score", counter)
        self.game_id += num_games
        LOGGER.debug('Moves in this training set:')
        LOGGER.debug("  Up: %d", np.sum(moves[:, Direction.UP.value]))
        LOGGER.debug("  Right: %d", np.sum(moves[:, Direction.RIGHT.value]))
        LOGGER.debug("  Down: %d", np.sum(moves[:, Direction.DOWN.value]))
        LOGGER.debug("  Left: %d", np.sum(moves[:, Direction.LEFT.value]))
        self.add_games_to_list(states=states, heads=heads, scores=scores,
                               ids=ids, moves=moves, generation=generation)

    def add_games_to_list(self,
                          *,
                          states: npt.NDArray[np.int32],
                          heads: npt.NDArray[np.bool8],
                          scores: npt.NDArray[np.int32],
                          ids: npt.NDArray[np.int32],
                          moves: npt.NDArray[np.float32],
                          generation: int,
                          make_histogram: bool = True) -> None:
        """Add new games to overall game list.

        Args:
            states (npt.NDArray[np.int32]): states for most recent play set
            heads (npt.NDArray[np.int32]): heads
            scores (npt.NDArray[np.int32]): scores
            ids (npt.NDArray[np.int32]): ids
            moves (npt.NDArray[np.int32]): moves
            generation (int): Generation number.
            make_histogram (bool): Make a histogram for this set of games.
        """
        _, indices = np.unique(ids, return_index=True)
        self.mean_score = np.mean(scores[indices])

        if make_histogram:
            self.gen_histogram(scores=scores[indices], generation=generation)

        self.game_states = np.concatenate(
            (self.game_states, states))
        self.game_heads = np.concatenate((self.game_heads, heads))
        self.game_scores = np.concatenate(
            (self.game_scores, scores))
        self.game_ids = np.concatenate((self.game_ids, ids))
        self.moves = np.concatenate((self.moves, moves))

    def trim_game_list(self) -> None:
        """Remove lowest scoring games from game list."""
        uni, indices = np.unique(self.game_ids, return_index=True)
        sorted_scores = np.sort(self.game_scores[indices])
        number_of_games = len(uni)
        purge_num = number_of_games - \
            NUM_TRAINING_GAMES-1 if number_of_games >= NUM_TRAINING_GAMES else 0

        # purge worst games or all games below 0 score
        # the problem with this method is that you get rid of too many at once
        # ideally I think we just want to purge the worst 500
        purge_score = max(sorted_scores[purge_num], 0) if purge_num > 0 else 0

        # get rid of low score games
        valid_idx = np.nonzero(self.game_scores > purge_score)

        self.game_ids = self.game_ids[valid_idx]
        self.game_scores = self.game_scores[valid_idx]
        self.game_states = self.game_states[valid_idx]
        self.moves = self.moves[valid_idx]
        self.game_heads = self.game_heads[valid_idx]

        # update tracking statistics
        uni, indices = np.unique(self.game_ids, return_index=True)
        number_of_games = len(uni)
        self.mean_score = np.mean(self.game_scores[indices])
        self.mean_scores.append(self.mean_score)
        self.games_used.append(number_of_games)
        self.max_scores.append(np.max(self.game_scores))
        try:
            LOGGER.info("Generation mean score: %02f -> %02f",
                        self.mean_scores[-2], self.mean_scores[-1])
            LOGGER.info("Generation maximum score: %02f -> %02f",
                        self.max_scores[-2], self.max_scores[-1])
        except IndexError:
            pass  # don't print previous score
        LOGGER.info("Games in training set: %d", self.games_used[-1])

    def print_size_info(self) -> None:
        """Print size of various members."""
        LOGGER.debug('class %d', get_size(self))
        LOGGER.debug('gameHeads %d', get_size(self.game_heads))
        LOGGER.debug('gameids %d', get_size(self.game_ids))
        LOGGER.debug('gameScores %d', get_size(self.game_scores))
        LOGGER.debug('gameStates %d', get_size(self.game_states))
        LOGGER.debug('moves %d', get_size(self.moves))
