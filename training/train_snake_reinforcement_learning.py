"""Reinforcement learning."""

import logging
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt

from training.helper import (GRID_X, GRID_Y, NUM_SELF_PLAY_GAMES,
                             NUM_TRAINING_GAMES, SAVE_INTERVAL, SCORE_PER_FOOD,
                             Timer, gen_histogram, get_size)
from training.neural_net import NeuralNetwork
from training.play_games import PlayBig
from training.trainer import train

LOGGER = logging.getLogger("terminal")


class TrainRL:
    """Reinforcement learning loop."""

    def __init__(self, minimum: int = 0) -> None:
        self.minimum = minimum
        self.game_states = np.zeros((0, GRID_Y, GRID_X), dtype=np.int32)
        self.game_heads = np.zeros((0, 4), dtype=np.bool8)
        self.game_scores = np.zeros((0,), dtype=np.int32)
        self.game_ids = np.zeros((0,), dtype=np.int32)
        self.moves = np.zeros((0, 4), dtype=np.float32)
        saves = list(Path("./media/saves").glob("*.ckpt"))
        self.neural_net = NeuralNetwork()
        self.best_neural_net = NeuralNetwork()
        self.generation = 0
        if len(saves) > 0:
            biggest_generation = max(
                [int(re.findall(r'\d+', save.name)[0]) for save in saves])
            biggest_path = Path(
                f"./media/saves/generation_{biggest_generation}.ckpt")
            self.neural_net.load(biggest_path)
            self.best_neural_net.load(biggest_path)
            self.generation = biggest_generation
            LOGGER.debug("Loaded checkpoint %d", biggest_generation)
        self.game_id = 0
        # mean score of the last set of games
        self.mean_generation_score: float = -SCORE_PER_FOOD
        # mean scores of each set of games
        self.mean_generation_scores: List[float] = []
        # mean score of all games in training set
        self.training_mean_scores: List[float] = []
        self.games_used: List[int] = []
        self.max_scores: List[int] = []
        self.times: List[float] = []

    def train(self) -> None:
        """Training loop."""
        generation = self.generation
        while 1:
            LOGGER.info("")
            LOGGER.info("")
            LOGGER.info("Generation %d", generation)
            with Timer("Generation") as t:
                self.play_n_games(
                    generation=generation, num_games=NUM_SELF_PLAY_GAMES)
                purge_num = max(len(np.unique(self.game_ids)) -
                                NUM_TRAINING_GAMES, NUM_SELF_PLAY_GAMES)
                self.trim_game_list(int(purge_num))
                self.neural_net = train(
                    generation, self.game_states, self.game_heads, self.moves, verbose=1)
            self.times.append(t.secs)
            self.gen_status_plot(generation)
            if generation % SAVE_INTERVAL == 0:
                gen_path = Path(f"./media/saves/generation_{generation}.ckpt")
                self.game_states.tofile(gen_path/"states.npy")
                self.game_heads.tofile(gen_path/"heads.npy")
                self.moves.tofile(gen_path/"moves.npy")

            generation += 1

    def gen_status_plot(self, generation: int) -> None:
        """Plot status.

        Args:
            generation (int): Generation number.
        """
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(self.training_mean_scores, label="Training")
        plt.plot(self.mean_generation_scores, label="Generation")
        plt.legend(loc="best")
        plt.title('Mean Score')

        ax1 = plt.subplot(3, 1, 2)
        color = "#69b3a2"
        ax1.plot(self.max_scores, color=color)
        ax1.set_ylabel('Max Score', color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax2 = ax1.twinx()
        ax2.plot(self.games_used)
        ax2.set_ylabel('Training Games')

        plt.subplot(3, 1, 3)
        plt.plot(self.times)
        plt.xlabel('Time(s)')

        plt.suptitle(f'Generation: {generation}')

        fig.savefig('media/pictures/mostrecent.png')
        plt.close()

    def play_n_games(self, *, generation: int, num_games: int = NUM_SELF_PLAY_GAMES) -> None:
        """Play one generation worth of snake games.

        Args:
            generation (int): Generation number.
            num_games (int, optional): Number of games to play. Defaults to NUM_SELF_PLAY_GAMES.
        """
        counter = 0
        state_l = []
        head_l = []
        scores_l = []
        ids_l = []
        moves_l = []

        # update minimum allowable score for this generation
        minimum_score = self.minimum if self.game_scores.size == 0 else self.game_scores.min()

        # play one set with last neural net to check if best neural net has been created
        states, heads, scores, ids, moves, count, performance = self.play_one_set_of_games(
            self.neural_net, minimum_score=minimum_score)
        self.mean_generation_score = performance
        state_l.append(states)
        head_l.append(heads)
        scores_l.append(scores)
        ids_l.append(ids)
        moves_l.append(moves)
        counter += count

        # get maximum score
        max_score = self.get_maximum_generation_score()

        # consider updating best neural net
        if performance > max_score:
            LOGGER.info("New best neural net created (%02f -> %02f).",
                        max_score, performance)
            self.best_neural_net = self.neural_net

        # run games until desired number of games above minimum score have been run
        while counter < num_games:
            LOGGER.info("%d games so far.", counter)
            states, heads, scores, ids, moves, num_added, _ = self.play_one_set_of_games(
                neural_net=self.best_neural_net, minimum_score=minimum_score)

            counter += num_added

            state_l.append(states)
            head_l.append(heads)
            scores_l.append(scores)
            ids_l.append(ids)
            moves_l.append(moves)

        self.add_games_to_list(states=np.concatenate(state_l),
                               heads=np.concatenate(head_l),
                               scores=np.concatenate(scores_l),
                               ids=np.concatenate(ids_l),
                               moves=np.concatenate(moves_l),
                               generation=generation)

    def get_maximum_generation_score(self) -> float:
        """Return maximum score atained by a training of the neural network.

        Returns:
            int: Maximum score.
        """
        try:
            return max(self.mean_generation_scores)
        except ValueError:
            return self.mean_generation_score

    def play_one_set_of_games(self, neural_net: NeuralNetwork, *, minimum_score: int) -> Tuple[npt.NDArray[np.int32],
                                                                                               npt.NDArray[np.bool8],
                                                                                               npt.NDArray[np.int32],
                                                                                               npt.NDArray[np.int32],
                                                                                               npt.NDArray[np.float32],
                                                                                               int,
                                                                                               float]:
        """Play one generation worth of snake games.

        Args:
            neural_net: (NeuralNetwork):
            generation (int): Generation number.

        Returns:
            (npt.NDArray[np.uint32]): game states
            (npt.NDArray[npt.bool8]): heads
            (npt.NDArray[np.int32]): scores
            (npt.NDArray[np.int32]): game ids
            (npt.NDArray[np.float32]): predictions
            (int): number of games
            (float): performance
        """
        spc = PlayBig(neural_network=neural_net)

        states, heads, scores, ids, moves, performance = spc.play_games(
            start_id=self.game_id, minimum_score=minimum_score, exploratory=True, best_generation_score=self.get_maximum_generation_score())

        counter = len(np.unique(ids))

        self.game_id += counter
        return states, heads, scores, ids, moves, counter, performance

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

        if make_histogram:
            gen_histogram(scores=scores[indices], generation=generation)

        self.game_states = np.concatenate(
            (self.game_states, states))
        self.game_heads = np.concatenate((self.game_heads, heads))
        self.game_scores = np.concatenate(
            (self.game_scores, scores))
        self.game_ids = np.concatenate((self.game_ids, ids))
        self.moves = np.concatenate((self.moves, moves))

    def trim_game_list(self, purge_num: int) -> None:
        """Remove lowest scoring games from game list."""
        uni, indices = np.unique(self.game_ids, return_index=True)
        sorted_scores = np.sort(self.game_scores[indices])
        number_of_games = len(uni)

        # purge worst games or all games below 0 score
        # the problem with this method is that you get rid of too many at once
        # ideally I think we just want to purge the worst 500
        purge_score = max(sorted_scores[purge_num], 0) if purge_num > 0 else 0

        # get rid of low score games
        valid_idx = np.nonzero(self.game_scores > purge_score)
        LOGGER.info("Keeping %d games out of %d", np.sum(
            sorted_scores > purge_score), len(sorted_scores))

        self.game_ids = self.game_ids[valid_idx]
        self.game_scores = self.game_scores[valid_idx]
        self.game_states = self.game_states[valid_idx]
        self.moves = self.moves[valid_idx]
        self.game_heads = self.game_heads[valid_idx]

        # update tracking statistics
        uni, indices = np.unique(self.game_ids, return_index=True)
        number_of_games = len(uni)
        self.mean_generation_scores.append(self.mean_generation_score)
        self.training_mean_scores.append(np.mean(self.game_scores[indices]))
        self.games_used.append(number_of_games)
        self.max_scores.append(np.max(self.game_scores))
        try:
            LOGGER.info("Training mean score: %02f -> %02f",
                        self.training_mean_scores[-2], self.training_mean_scores[-1])
        except IndexError:
            pass  # don't print previous score

    def print_size_info(self) -> None:
        """Print size of various members."""
        LOGGER.debug('class %d', get_size(self))
        LOGGER.debug('gameHeads %d', get_size(self.game_heads))
        LOGGER.debug('gameids %d', get_size(self.game_ids))
        LOGGER.debug('gameScores %d', get_size(self.game_scores))
        LOGGER.debug('gameStates %d', get_size(self.game_states))
        LOGGER.debug('moves %d', get_size(self.moves))
