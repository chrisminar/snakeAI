# todo list
# test for timeout
# experiemnt with different optimizers

#################
### soon list ###
#################
# try to progressively update nn, not reinit every time?
# there is some sort of bug that makes the game end too early

#################
### done list ###
#################
# figure out how stop doing additional epochs once the training plateaus
# stop training if accuracy and val accuracy are not improving
# make histogram of game scores
# try setting score_per_move to -1
# save nn on generation completion
# possible bug with gamelength
# fixed
# current scheme results in very low number of games played after the mean score increases.
# should change the game trimming to be a bit more complex, e.g. always keep ~10k games and remove the worst ones
# need to take the highest valid class output, not random if highest is invalid. Can do this in the nn itself.
# change trim game list such that the highest mean score is remembered. Every time you pass Prune games below highest mean every pass.
# possible failure of trim game states after too many runs (numer of samples constantly increases)
# need to rework. Currently takes newest game id - oldest game id. it should count the total number of game ids
# possible stagnation
# around 450
# try training on more epochs? -- seems to stop benefitting around 2
# try init snake in middle (not 0,0)
# moved to 1,1

###################
### FUTURE LIST ###
###################
# stuff to do someday
# do network pruning
# track/understand what the neural network is doing
# gui
# pause/play

import numpy as np
from matplotlib import pyplot as plt

from globalVar import Globe as globe
from neuralNet import NeuralNetwork
from selfPlay import SelfPlay
from trainer import Trainer


class TrainRL:
    def __init__(self) -> None:
        self.gameStates = np.zeros((0, globe.GRID_X, globe.GRID_Y))
        self.gameHeads = np.zeros((0, 4))
        self.gameScores = np.zeros((0,))
        self.gameIDs = np.zeros((0,))
        self.moves = np.zeros((0, 4))
        self.nn = NeuralNetwork()
        self.gameID = 0
        self.meanScore = 0
        self.meanScores = []
        self.gamesUsed = []
        self.maxScores = []

    def train(self) -> None:
        generation = 0
        while 1:
            print('')
            print('######################')
            print('### Generation {}###'.format(generation))
            print('######################')
            numberOfGames = len(np.unique(self.gameIDs))
            n = globe.NUM_TRAINING_GAMES - numberOfGames
            if n > globe.NUM_SELF_PLAY_GAMES:  # if we need many more games, play many more games
                pass
            else:  # if we already have a lot of games, use default amount
                n = globe.NUM_SELF_PLAY_GAMES
            print('num games to play {}'.format(n))
            self.self_play(self.nn, generation, n)
            self.network_trainer(generation)
            #print(generation, globe.getsize(self.nn))
            self.gen_status_plot(generation)
            generation += 1

    # save plot that shows status of training
    def gen_status_plot(self, generation: int) -> None:
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(self.meanScores)
        plt.title('Mean Score')

        plt.subplot(3, 1, 2)
        plt.plot(self.maxScores)
        plt.title('Max Score')

        plt.subplot(3, 1, 3)
        plt.plot(self.gamesUsed)
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
    def self_play(self, nn: NeuralNetwork, generation: int, num_games: int = globe.NUM_SELF_PLAY_GAMES) -> None:
        spc = SelfPlay(nn)
        states, heads, scores, ids, moves = spc.play_games(
            generation, self.gameID, num_games)
        self.gameID += num_games
        print('Moves in this training set:  Up: ', np.sum(moves[:, 0]), ', Right: ', np.sum(
            moves[:, 1]), ', Down: ', np.sum(moves[:, 2]), ', Left: ', np.sum(moves[:, 3]))
        self.add_games_to_list(states, heads, scores, ids, moves, generation)
        self.trim_game_list()

    # operates on:
        # last 10000 games of self play
    # outputs:
        # new neural network
    def network_trainer(self, generation: int) -> None:
        del self.nn
        self.nn = NeuralNetwork()
        trn = Trainer(self.nn)
        trn.train(generation, self.gameStates, self.gameHeads, self.moves)

    def add_games_to_list(self,
                          states: npt.NDArray[np.int32],
                          heads: npt.NDArray[np.int32],
                          scores: npt.NDArray[np.int32],
                          ids: npt.NDArray[np.int32],
                          moves: npt.NDArray[np.int32],
                          generation: int) -> None:
        uni, indices = np.unique(ids, return_index=True)
        self.meanScore = np.mean(scores[indices])

        # get rid of bad scores
        self.gen_histogram(scores[indices], generation)

        if self.meanScore > 50:
            cutoff = self.meanScore
        else:
            cutoff = 100
        validIdx = scores > cutoff
        ids = ids[validIdx]
        scores = scores[validIdx]
        states = states[validIdx]
        moves = moves[validIdx]
        heads = heads[validIdx]

        uni, indices = np.unique(ids, return_index=True)
        self.meanScoreAfter = np.mean(scores[indices])

        self.gameStates = np.concatenate((self.gameStates, states))
        self.gameHeads = np.concatenate((self.gameHeads, heads))
        self.gameScores = np.concatenate((self.gameScores, scores))
        self.gameIDs = np.concatenate((self.gameIDs, ids))
        self.moves = np.concatenate((self.moves, moves))

    def trim_game_list(self) -> None:
        # get rid of no-score games
        validIdx = np.nonzero(self.gameScores > -50)
        self.gameIDs = self.gameIDs[validIdx]
        self.gameScores = self.gameScores[validIdx]
        self.gameStates = self.gameStates[validIdx]
        self.moves = self.moves[validIdx]
        self.gameHeads = self.gameHeads[validIdx]

        uni, indices = np.unique(self.gameIDs, return_index=True)
        numberOfGames = len(uni)

        count = 0
        badIdx = []
        idx = 0
        lowScore = np.min(self.gameScores)

        overallMean = np.mean(self.gameScores[indices])

        # if you have too many games, get rid of the worst
        if numberOfGames > globe.NUM_TRAINING_GAMES:
            purgeNum = numberOfGames - globe.NUM_TRAINING_GAMES
        else:
            purgeNum = 0
        while count < purgeNum:
            startIdx = idx

            # check if current game is bad
            if self.gameScores[idx] < overallMean:
                count += 1
                bad = True
            else:
                bad = False

            # seek to next game and invalidate if nessiscary
            flag = True
            while flag:
                if bad:
                    badIdx.append(idx)
                idx += 1
                if idx >= len(self.gameScores):
                    count = purgeNum
                    flag = False
                elif self.gameIDs[idx] != self.gameIDs[startIdx]:
                    flag = False

        mask = np.ones(len(self.gameIDs), dtype=bool)
        mask[badIdx] = False
        self.gameIDs = self.gameIDs[mask]
        self.gameScores = self.gameScores[mask]
        self.gameStates = self.gameStates[mask]
        self.moves = self.moves[mask]
        self.gameHeads = self.gameHeads[mask]

        oldNumberOfGames = numberOfGames
        uni, indices = np.unique(self.gameIDs, return_index=True)
        numberOfGames = len(uni)
        overallMeanAfter = np.mean(self.gameScores[indices])

        # update statistics
        self.meanScores.append(self.meanScore)
        self.gamesUsed.append(numberOfGames)
        self.maxScores.append(np.max(self.gameScores))
        print('Generation mean score in  before/after purge: {}/{}.\n Over mean score before/after purge {}/{}.\n Best score of {} in {} games.'.format(self.meanScore,
                                                                                                                                                        self.meanScoreAfter,
                                                                                                                                                        overallMean,
                                                                                                                                                        overallMeanAfter,
                                                                                                                                                        self.maxScores[
                                                                                                                                                            -1],
                                                                                                                                                        self.gamesUsed[-1]))

    def size_info(self) -> None:
        print('self',       globe.get_size(self))
        print('gameHeads',  globe.get_size(self.gameHeads))
        print('gameids',    globe.get_size(self.gameIDs))
        print('gameScores', globe.get_size(self.gameScores))
        print('gameStates', globe.get_size(self.gameStates))
        print('moves',      globe.get_size(self.moves))
