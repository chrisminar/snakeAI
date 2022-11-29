"""Test trainer."""

from training.trainer import train


# TODO needs work
def test_train() -> None:
    """Test trainer."""
    game_states = []  # type:ignore
    heads = []  # type:ignore
    move_predictions = []  # type:ignore
    _ = train(generation=0, game_states=game_states,   # type:ignore
              heads=heads, move_predictions=move_predictions)  # type:ignore
