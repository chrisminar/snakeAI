"""Test trainer."""

from trainer import train


# TODO needs work
def test_train() -> None:
    """Test trainer."""
    game_states = []
    heads = []
    move_predictions = []
    _ = train(generation=0, game_states=game_states,
              heads=heads, move_predictions=move_predictions)
