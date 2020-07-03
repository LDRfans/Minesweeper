import numpy as np
import os
import random
import matplotlib.pyplot as plt

from MineSweeper import MineSweeper
from BP_Neural_Network import Neural_Network

# set the training and testing rounds
num_train_rounds = 50000

dim_1_sel = [4]
dim_2_sel = [4]
num_mines_sel = [3]

rounds = []
winning_rates = []
for round in range(num_train_rounds):
    if(round%100==0):
        rounds.append(round)

for b in range(1):
    # set the attributes of the game
    dim_1 = dim_1_sel[b]
    dim_2 = dim_2_sel[b]
    num_mines = np.random.randint(2,4)

    # count the number of uncovered tiles
    count = dim_1*dim_2

    # calculate the number of input neurons = (# of tile) * (# of possiblities) + 1
    input_size = dim_1*dim_2*10+1
    # set the number of output neurons = # of tiles
    output_size = dim_1*dim_2 
    # calculate the number of hidden neurons = sqrt[(out+2)in] + 2sqrt[in/(out+2)])
    hidden_size = int(np.sqrt((output_size+2)*input_size)+2*np.sqrt(input_size/(output_size+2)))
    # create a BPNN  
    BPNN = Neural_Network(input_size, hidden_size, output_size)

    num_trainng_succ = 0

    winning_rates_b = []
    # train the model
    for round in range(num_train_rounds):
        if(round%100==0):
            winning_rates_b.append(num_trainng_succ/100)
            num_trainng_succ = 0
        # create a minesweeper
        game = MineSweeper(dim_1, dim_2, num_mines)
        # start from the lefttop tile
        game.selectCell([0,0])
        while not game.gameOver:
            # reset the input layer of BPNN
            BPNN.input = np.zeros(input_size)
            # traverse game state to get the input layer
            for i in range(game.dim1):
                for j in range(game.dim2):
                    if np.isnan(game.state[i,j]):
                        BPNN.input[(i*game.dim2+j)*10+9] = 1
                    else:
                        BPNN.input[int((i*game.dim2+j)*10+game.state[i,j])] = 1
                        count -= 1
            # the last input indicate the ratio between mines and covered tiles
            if count > 0:
                BPNN.input[game.totalCells*10] = num_mines/count

            # let BPNN feeds forward
            BPNN.feedforward()
            # do back propagation using true distribution of mines as true value output
            BPNN.output_true_value = game.mines.flatten()
            BPNN.backprop()

            # add coordinate to output
            output_with_coor = np.zeros([game.totalCells,2])
            for i in range(game.totalCells):
                output_with_coor[i][0] = BPNN.output[i]
                output_with_coor[i][1] = i
            
            # find the minimum probability output
            output_with_coor = output_with_coor[output_with_coor[:,0].argsort()]
            # to iterate the output
            iterator = 0
            # check whether it is covered to break the loop
            while True:
                min_prob_out = output_with_coor[iterator,1]
                next_tile_coor = [int(min_prob_out//game.dim2),int(min_prob_out%game.dim2)] 
                if np.isnan(game.state[next_tile_coor[0],next_tile_coor[1]]):
                    break
                iterator += 1
            
            # prob at the next chosen tile
            game.selectCell(next_tile_coor)
            # game.selectCell(random.choice(valid_tiles))
        if game.victory:
            num_trainng_succ += 1
    winning_rates.append(winning_rates_b)

plt.figure()
plt.plot(rounds,winning_rates[0],label="4x4,3")

plt.title("Winning Rate Along With Training(for 4x4 board and 3 mines)")
plt.xlabel("Training Rounds")
plt.ylabel("Winning Rate")
plt.legend(loc=1)

plt.savefig("fig_2.png")