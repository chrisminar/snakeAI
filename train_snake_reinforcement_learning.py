# TODO test for timeout
# TODO experiemnt with different optimizers
# TODO try to progressively update nn, not reinit every time?
# TODO there is some sort of bug that makes the game end too early
# TODO do network pruning
# TODO track/understand what the neural network is doing
# TODO gui
# TODO pause/play
"""Reinforcement learning."""

from typing import List

import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt

from helper import (GRID_X, GRID_Y, NUM_SELF_PLAY_GAMES, NUM_TRAINING_GAMES,
                    get_size)
from neural_net import NeuralNetwork
from play_games import PlayGames
from trainer import Trainer


class TrainRL:
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
        generation = 0
        while 1:
            print('')
            print('######################')
            print('### Generation {}###'.format(generation))
            print('######################')
            number_of_games = len(np.unique(self.game_ids))
            n = NUM_TRAINING_GAMES - number_of_games
            if n > NUM_SELF_PLAY_GAMES:  # if we need many more games, play many more games
                pass
            else:  # if we already have a lot of games, use default amount
                n = NUM_SELF_PLAY_GAMES
            print('num games to play {}'.format(n))
            self.self_play(self.neural_net, generation, n)
            self.network_trainer(generation)
            self.gen_status_plot(generation)
            generation += 1

    # save plot that shows status of training
    def gen_status_plot(self, generation: int) -> None:
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

        plt.suptitle('Generation: {}'.format(generation))

        fig.savefig('mostrecent.png')
        plt.close()

    def gen_histogram(self, unique: npt.NDArray[np.int32], generation: int) -> None:
        fig = plt.figure()
        plt.title('Generation {}'.format(generation))
        plt.hist(unique, bins=[0, 100, 200, 300, 400, 500, 600,
                 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600])
        fig.savefig('hists/generation_{}.png'.format(generation))
        plt.close()

    # operates on:
        # neural network
    # outputs:
        # 2000 game outputs
    def self_play(self, nn: NeuralNetwork, generation: int, num_games: int = NUM_SELF_PLAY_GAMES) -> None:
        spc = PlayGames(nn)
        states, heads, scores, ids, moves = spc.play_games(
            self.game_id, num_games)
        self.game_id += num_games
        print('Moves in this training set:  Up: ', np.sum(moves[:, 0]), ', Right: ', np.sum(
            moves[:, 1]), ', Down: ', np.sum(moves[:, 2]), ', Left: ', np.sum(moves[:, 3]))
        self.add_games_to_list(states, heads, scores, ids, moves, generation)
        self.trim_game_list()

    # operates on:
        # last 10000 games of self play
    # outputs:
        # new neural network
    def network_trainer(self, generation: int) -> None:
        del self.neural_net
        self.neural_net = NeuralNetwork()
        trn = Trainer(self.neural_net)
        trn.train(generation, self.game_states, self.game_heads, self.moves)

    def add_games_to_list(self,
                          states: npt.NDArray[np.int32],
                          heads: npt.NDArray[np.int32],
                          scores: npt.NDArray[np.int32],
                          ids: npt.NDArray[np.int32],
                          moves: npt.NDArray[np.int32],
                          generation: int) -> None:
        uni, indices = np.unique(ids, return_index=True)
        self.mean_score = np.mean(scores[indices])

        # get rid of bad scores
        self.gen_histogram(scores[indices], generation)

        if self.mean_score > 50:
            cutoff = self.mean_score
        else:
            cutoff = 100
        valid_idx = scores > cutoff
        ids = ids[valid_idx]
        scores = scores[valid_idx]
        states = states[valid_idx]
        moves = moves[valid_idx]
        heads = heads[valid_idx]

        uni, indices = np.unique(ids, return_index=True)
        self.mean_score_after = np.mean(scores[indices])

        self.game_states = np.concatenate((self.game_states, states))
        self.game_heads = np.concatenate((self.game_heads, heads))
        self.game_scores = np.concatenate((self.game_scores, scores))
        self.game_ids = np.concatenate((self.game_ids, ids))
        self.moves = np.concatenate((self.moves, moves))

    def trim_game_list(self) -> None:
        # get rid of no-score games
        valid_idx = np.nonzero(self.game_scores > -50)
        self.game_ids = self.game_ids[valid_idx]
        self.game_scores = self.game_scores[valid_idx]
        self.game_states = self.game_states[valid_idx]
        self.moves = self.moves[valid_idx]
        self.game_heads = self.game_heads[valid_idx]

        uni, indices = np.unique(self.game_ids, return_index=True)
        number_of_games = len(uni)

        count = 0
        bad_idx = []
        idx = 0
        low_score = np.min(self.game_scores)

        overall_mean = np.mean(self.game_scores[indices])

        # if you have too many games, get rid of the worst
        if number_of_games > NUM_TRAINING_GAMES:
            purge_num = number_of_games - NUM_TRAINING_GAMES
        else:
            purge_num = 0
        while count < purge_num:
            start_idx = idx

            # check if current game is bad
            if self.game_scores[idx] < overall_mean:
                count += 1
                bad = True
            else:
                bad = False

            # seek to next game and invalidate if nessiscary
            flag = True
            while flag:
                if bad:
                    bad_idx.append(idx)
                idx += 1
                if idx >= len(self.game_scores):
                    count = purge_num
                    flag = False
                elif self.game_ids[idx] != self.game_ids[start_idx]:
                    flag = False

        mask = np.ones(len(self.game_ids), dtype=bool)
        mask[bad_idx] = False
        self.game_ids = self.game_ids[mask]
        self.game_scores = self.game_scores[mask]
        self.game_states = self.game_states[mask]
        self.moves = self.moves[mask]
        self.game_heads = self.game_heads[mask]

        old_number_of_games = number_of_games
        uni, indices = np.unique(self.game_ids, return_index=True)
        number_of_games = len(uni)
        overall_mean_after = np.mean(self.game_scores[indices])

        # update statistics
        self.mean_scores.append(self.mean_score)
        self.games_used.append(number_of_games)
        self.max_scores.append(np.max(self.game_scores))
        print('Generation mean score in  before/after purge: {}/{}.\n Over mean score before/after purge {}/{}.\n Best score of {} in {} games.'.format(self.mean_score,
                                                                                                                                                        self.mean_score_after,
                                                                                                                                                        overall_mean,
                                                                                                                                                        overall_mean_after,
                                                                                                                                                        self.max_scores[
                                                                                                                                                            -1],
                                                                                                                                                        self.games_used[-1]))

    def size_info(self) -> None:
        print('self',       get_size(self))
        print('gameHeads',  get_size(self.game_heads))
        print('gameids',    get_size(self.game_ids))
        print('gameScores', get_size(self.game_scores))
        print('gameStates', get_size(self.game_states))
        print('moves',      get_size(self.moves))
