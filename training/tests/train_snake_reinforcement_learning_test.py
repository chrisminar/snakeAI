"""Test Reinforcement learning snake."""

import numpy as np

from helper import GRID_X, GRID_Y, NUM_TRAINING_GAMES
from neural_net import NeuralNetwork
from play_games import PlayGames
from train_snake_reinforcement_learning import TrainRL


def test_init() -> None:
    """Test trainer initialization works."""
    TrainRL()


def test_add_games_to_list() -> None:
    """Add games to list works."""
    trainer = TrainRL()
    neural_network = NeuralNetwork()
    spc = PlayGames(neural_network)
    states, heads, scores, ids, moves = spc.play_games(0, 5)
    trainer.add_games_to_list(states=states, heads=heads,
                              scores=scores, ids=ids, moves=moves, generation=0)
    assert np.max(trainer.game_ids) == 4
    assert trainer.game_states.shape[0] == trainer.game_ids.shape[0], 'bad gameid shape'
    assert trainer.game_states.shape[0] == trainer.game_scores.shape[0], 'bad score shape'
    assert trainer.game_states.shape[0] == trainer.moves.shape[0], 'bad moves shape'


def test_trim_game_list() -> None:
    """Trim games works."""
    num_games = NUM_TRAINING_GAMES
    moves_per_game = 3
    n_moves = num_games*moves_per_game
    moves = np.zeros((n_moves, 4), dtype=np.float32)
    states = np.zeros((n_moves, GRID_X, GRID_Y), dtype=np.int32)
    scores = np.zeros((n_moves,), dtype=np.int32)
    game_ids = np.zeros((moves_per_game,), dtype=np.int32)
    heads = np.ones((n_moves, 4), dtype=np.bool8)
    for i in range(1, num_games):
        game_ids = np.concatenate(
            [game_ids, np.zeros((moves_per_game,)) + i])
    trainer = TrainRL()
    trainer.add_games_to_list(states=states, heads=heads,
                              scores=scores, ids=game_ids, moves=moves, generation=0)

    offset = game_ids[-1]+1
    game_ids = np.zeros((moves_per_game,)) + offset
    for i in range(1, num_games):
        game_ids = np.concatenate(
            [game_ids, np.zeros(moves_per_game,) + i + offset])
    trainer.add_games_to_list(states=states, heads=heads,
                              scores=scores+100, ids=game_ids, moves=moves, generation=0)
    trainer.trim_game_list()
    assert trainer.game_states.shape[0] == NUM_TRAINING_GAMES * \
        moves_per_game, 'Gamestates trim unsuccessful'
    assert trainer.moves.shape[0] == NUM_TRAINING_GAMES * \
        moves_per_game, 'Moves trim unsuccessful'
    assert trainer.game_scores.shape[0] == NUM_TRAINING_GAMES * \
        moves_per_game, 'Scores trim unsuccessful'
    assert trainer.game_ids.shape[0] == NUM_TRAINING_GAMES * \
        moves_per_game, 'game ids trim unsuccessful'
    assert np.min(trainer.game_scores) == 100, 'min scores not removed'
