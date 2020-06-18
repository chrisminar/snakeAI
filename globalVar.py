class Globe():
  GRID_X = 5                # x grid size of snake game
  GRID_Y = 5                # y grid size of snake game
  NUM_SELF_PLAY_GAMES = 500 # number of self play games to play
  NUM_TRAINING_GAMES = 2000 # number of self play games to train on
  MOMENTUM = 0.9
  BATCH_SIZE = 64
  EPOCHS = 5
  TIMEOUT = 100

  SCORE_PER_FOOD = 100