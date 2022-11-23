# TODO test for timeout
# TODO experiemnt with different optimizers
# TODO try to progressively update nn, not reinit every time?
# TODO there is some sort of bug that makes the game end too early
# TODO do network pruning
# TODO track/understand what the neural network is doing
# TODO gui
# TODO pause/play
"""Reinforcement learning."""

from typing import List, Sequence

import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt

from helper import (GRID_X, GRID_Y, NUM_SELF_PLAY_GAMES, NUM_TRAINING_GAMES,
                    get_size)
from neural_net import NeuralNetwork
from play_games import PlayGames
from trainer import Trainer


class TrainRL:
    """Reinforcement learning loop."""

    def __init__(self) -> None:
        self.game_states = np.zeros((0, GRID_X, GRID_Y))
        self.game_heads = np.zeros((0, 4))
        self.game_scores = np.zeros((0,))
        self.game_ids = np.zeros((0,))
        self.moves = np.zeros((0, 4))
        self.neural_net = NeuralNetwork()
        self.game_id = 0
        self.mean_score = 0
        self.mean_scores: List[int] = []
        self.games_used: List[int] = []
        self.max_scores: List[int] = []

    def train(self) -> None:
        """Training loop."""
        generation = 0
        while 1:
            print('')
            print('######################')
            print('### Generation {}###'.format(generation))
            print('######################')
            n = NUM_TRAINING_GAMES - len(np.unique(self.game_ids))
            if n > NUM_SELF_PLAY_GAMES:  # if we need many more games, play many more games
                pass
            else:  # if we already have a lot of games, use default amount
                n = NUM_SELF_PLAY_GAMES
            print(f'num games to play {n}')
            self.play_one_generation_of_games(self.neural_net, generation, n)
            self.trim_game_list()
            self.network_trainer(generation)
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

        fig.savefig('mostrecent.png')
        plt.close()

    def gen_histogram(self, scores: Sequence[np.int32], generation: int) -> None:
        """Plot histogram of game scores.

        Args:
            unique (npt.NDArray[np.int32]): Game scores
            generation (int): Most recent generation.
        """
        fig = plt.figure()
        plt.title(f'Generation {generation}')
        plt.hist(scores, bins=np.arange(start=0, stop=1700, step=100))
        fig.savefig(f'hists/generation_{generation}.png')
        plt.close()

    def play_one_generation_of_games(self, neural_net: NeuralNetwork, *, generation: int, num_games: int = NUM_SELF_PLAY_GAMES) -> None:
        """Play one generation worth of snake games.

        Args:
            neural_net (NeuralNetwork): Neural network to play games with.
            generation (int): Generation number.
            num_games (int, optional): Number of games to play. Defaults to NUM_SELF_PLAY_GAMES.
        """
        spc = PlayGames(neural_net)
        states, heads, scores, ids, moves = spc.play_games(
            self.game_id, num_games)
        self.game_id += num_games
        print('Moves in this training set:  Up: ', np.sum(moves[:, 0]), ', Right: ', np.sum(
            moves[:, 1]), ', Down: ', np.sum(moves[:, 2]), ', Left: ', np.sum(moves[:, 3]))
        self.add_games_to_list(states, heads, scores, ids, moves, generation)

    def network_trainer(self, generation: int) -> None:
        """Train new neural network on training data.

        Args:
            generation (int): Generation number.
        """
        trn = Trainer()
        self.neural_net = trn.train(
            generation, self.game_states, self.game_heads, self.moves)

    def add_games_to_list(self,
                          states: npt.NDArray[np.int32],
                          heads: npt.NDArray[np.int32],
                          scores: npt.NDArray[np.int32],
                          ids: npt.NDArray[np.int32],
                          moves: npt.NDArray[np.int32],
                          generation: int) -> None:
        """Add new games to overall game list.

        Args:
            states (npt.NDArray[np.int32]): states for most recent play set
            heads (npt.NDArray[np.int32]): heads
            scores (npt.NDArray[np.int32]): scores
            ids (npt.NDArray[np.int32]): ids
            moves (npt.NDArray[np.int32]): moves
            generation (int): Generation number.
        """
        _, indices = np.unique(ids, return_index=True)
        self.mean_score = np.mean(scores[indices])

        self.gen_histogram(scores[indices], generation)

        # get rid of low scoring games
        cutoff = self.mean_score if self.mean_scores > 50 else 100
        valid_idx = scores > cutoff
        ids = ids[valid_idx]
        scores = scores[valid_idx]
        states = states[valid_idx]
        moves = moves[valid_idx]
        heads = heads[valid_idx]

        self.game_states = np.concatenate((self.game_states, states))
        self.game_heads = np.concatenate((self.game_heads, heads))
        self.game_scores = np.concatenate((self.game_scores, scores))
        self.game_ids = np.concatenate((self.game_ids, ids))
        self.moves = np.concatenate((self.moves, moves))

    def trim_game_list(self) -> None:
        """Remove lowest scoreing games from game list."""
        score = self.mean_score
        uni, indices = np.unique(self.game_ids, return_index=True)
        sorted_scores = np.sort(self.game_states[indices])
        number_of_games = len(uni)
        purge_num = number_of_games - \
            NUM_SELF_PLAY_GAMES if number_of_games > NUM_TRAINING_GAMES else 0

        # purge worst games or all games below 0 score
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
        # print('Generation mean score in  before/after purge: {}/{}.\n Over mean score before/after purge {}/{}.\n Best score of {} in {} games.'.format(self.mean_score,
        #                                                                                                                                                self.mean_score_after,
        #                                                                                                                                                overall_mean,
        #                                                                                                                                                overall_mean_after,
        #                                                                                                                                                self.max_scores[
        #                                                                                                                                                    -1],
        #                                                                                                                                                self.games_used[-1]))

    def print_size_info(self) -> None:
        """Print size of various members."""
        print('self',       get_size(self))
        print('gameHeads',  get_size(self.game_heads))
        print('gameids',    get_size(self.game_ids))
        print('gameScores', get_size(self.game_scores))
        print('gameStates', get_size(self.game_states))
        print('moves',      get_size(self.moves))
