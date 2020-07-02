import numpy as np

from Naive_Linear import Naive_Linear
from MineSweeper import MineSweeper

def naive_play():
    turn = 0
    game = MineSweeper()
    # init: reveal the corner
    game.selectCell([0,0])
    # play the game
    while(not game.gameOver):
        turn += 1
        print(game.state)
        naive_player = Naive_Linear(game.state, 8, 8, 10)
        min = [0, 0, naive_player.average_prob*8*9]
        # find the min_prob tile
        prob_matrix = naive_player.travel()
        print(prob_matrix)
        if game.victory == True:    break
        for i in range(8):
            for j in range(8):
                if np.isnan(game.state[i,j]):  # must be nan
                    if prob_matrix[i,j] <= min[2]:
                        min[0] = i
                        min[1] = j
                        min[2] = prob_matrix[i,j]
        if min[0] == 0 and min[1] == 0:
            # choose a random tile?
            print("Random...?")
        print(min)
        game.selectCell([min[0],min[1]])
    return turn

# print(game.mines)
