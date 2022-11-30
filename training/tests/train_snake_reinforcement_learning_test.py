"""Test Reinforcement learning snake."""

import numpy as np

from training.helper import (GRID_X, GRID_Y, NUM_SELF_PLAY_GAMES,
                             NUM_TRAINING_GAMES)
from training.neural_net import NeuralNetwork
from training.play_games import PlayGames
from training.train_snake_reinforcement_learning import TrainRL


def test_init() -> None:
    """Test trainer initialization works."""
    TrainRL()


def test_add_games_to_list() -> None:
    """Add games to list works."""
    trainer = TrainRL()
    neural_network = NeuralNetwork()
    spc = PlayGames(neural_network)
    states, heads, scores, ids, moves = spc.play_games(num_games=5)
    trainer.add_games_to_list(states=states, heads=heads,
                              scores=scores, ids=ids, moves=moves, generation=0)
    assert np.max(trainer.game_ids) < 5
    assert 0 <= np.min(trainer.game_ids)
    assert trainer.game_states.shape[0] == trainer.game_ids.shape[
        0] == trainer.game_scores.shape[0] == trainer.moves.shape[0]


def test_trim_game_list_1() -> None:
    """Trim game with 0s and 100s removes all 0s."""
    num_games = NUM_TRAINING_GAMES
    moves_per_game = 3
    n_moves = num_games*moves_per_game

    # initialize dummy games
    moves = np.zeros((n_moves, 4), dtype=np.float32)
    states = np.zeros((n_moves, GRID_X, GRID_Y), dtype=np.int32)
    scores = np.zeros((n_moves,), dtype=np.int32)
    game_ids = np.zeros((moves_per_game,), dtype=np.int32)
    heads = np.ones((n_moves, 4), dtype=np.bool8)
    for i in range(1, num_games):
        game_ids = np.concatenate(
            [game_ids, np.zeros((moves_per_game,)) + i])

    # add games to list will keep all games because their scores are all the same
    trainer = TrainRL()
    trainer.add_games_to_list(states=states, heads=heads,
                              scores=scores, ids=game_ids, moves=moves, generation=0)

    # add a bunch more games of higher score, again add games to list will keep all games
    offset = game_ids[-1]+1
    game_ids = np.zeros((moves_per_game,)) + offset
    for i in range(1, num_games):
        game_ids = np.concatenate(
            [game_ids, np.zeros(moves_per_game,) + i + offset])
    trainer.add_games_to_list(states=states, heads=heads,
                              scores=scores+100, ids=game_ids, moves=moves, generation=1)

    # lowest games will be removed
    trainer.trim_game_list()

    # all arrays have the correct size after trimming
    assert NUM_TRAINING_GAMES * \
        moves_per_game == trainer.game_states.shape[0] == trainer.moves.shape[
            0] == trainer.game_scores.shape[0] == trainer.game_ids.shape[0]
    # low scores correctly purged
    assert np.min(trainer.game_scores) == 100


def test_trim_game_list_2() -> None:
    """Trim games with np.arrange removes the lowest n games."""
    num_games = NUM_TRAINING_GAMES
    moves_per_game = 3
    n_moves = num_games*moves_per_game

    # initialize dummy games
    moves = np.zeros((n_moves, 4), dtype=np.float32)
    states = np.zeros((n_moves, GRID_X, GRID_Y), dtype=np.int32)
    game_ids = np.zeros((moves_per_game,), dtype=np.int32)
    heads = np.ones((n_moves, 4), dtype=np.bool8)
    for i in range(1, num_games):
        game_ids = np.concatenate(
            [game_ids, np.zeros((moves_per_game,)) + i])
    scores = game_ids.copy()

    # add games to trainer artificially because add games to list would purge values
    trainer = TrainRL()
    trainer.moves = moves
    trainer.game_states = states
    trainer.game_ids = game_ids
    trainer.game_heads = heads
    trainer.game_scores = scores

    # lowest NUM_SELF_PLAY_GAMES games will be removed
    trainer.trim_game_list()

    # all arrays have the correct size after trimming
    assert trainer.game_states.shape[0] == trainer.moves.shape[
        0] == trainer.game_scores.shape[0] == trainer.game_ids.shape[0]
    # low scores correctly purged
    assert np.min(trainer.game_scores) == NUM_SELF_PLAY_GAMES + 1
    assert np.min(trainer.game_ids) == NUM_SELF_PLAY_GAMES + 1
