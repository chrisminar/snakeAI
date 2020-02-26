

class globe():
  GRID_X = 10                # x grid size of snake game
  GRID_Y = 10                # y grid size of snake game
  NUM_EVALUATION_GAMES = 400 # number of evaluation games to play
  NUM_SELF_PLAY_GAMES = 2000 # number of self play games to play
  NUM_TRAINING_GAMES = 20000 # number of self play games to train on
  MOMENTUM = 0.9
  BATCH_SIZE = 64
  EPOCHS=5