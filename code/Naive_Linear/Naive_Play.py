import numpy as np

from Naive_Linear import Naive_Linear
from MineSweeper import MineSweeper

dim_1 = 4
dim_2 = 4
mines = 3

def naive_play(weights, bias):
    global min
    game = MineSweeper()
    # init: reveal the corner
    game.selectCell([0,0])
    # play the game
    while(not game.gameOver):
        # print(game.state)
        naive_player = Naive_Linear(game.state, game.neighbors, dim_1, dim_2, mines, weights, bias)
        # min = [x, y, prob, win, states]
        min = [0, 0, 0, 0, game.state]
        # find the min_prob tile
        prob_matrix = naive_player.travel()
        # print(prob_matrix)
        if game.victory == True:
            min[3] = 1
            return min
        for i in range(dim_1):
            for j in range(dim_2):
                if np.isnan(game.state[i,j]): # must be nan
                    if min[2]==0 or prob_matrix[i,j] <= min[2]:
                        min[0] = i
                        min[1] = j
                        min[2] = prob_matrix[i,j]
        min[4] = naive_player.input
        game.selectCell([min[0],min[1]])
        # print(min)
    # print(game.mines)
    return min


# game = MineSweeper()
# game.selectCell([0,0])
# print(game.neighbors)
# print(game.mines)

# weights = np.random.randn(9) 
# print(naive_play(weights, 1))