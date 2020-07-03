import numpy as np
import os
import random

from MineSweeper import MineSweeper
from BP_Neural_Network import Neural_Network

# set the attributes of the game
dim_1 = 8
dim_2 = 8
num_mines = 10

# set the training and testing rounds
num_train_rounds = 1000
num_test_rounds = 1000

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
# train the model
for round in range(num_train_rounds):
    # num_mines = np.random.randint(2,4)
    if(round%100==0):
        print(round,num_trainng_succ/100)
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
print(num_trainng_succ)
# fo_0 = open("attributes.txt", "w")
# fo_0.write(str(num_train_rounds)+" "+str(num_test_rounds)+" ")
# fo_0.close()

# fo_1 = open("temp_weights_in_h.txt", "w")
# for i in BPNN.weights_in_h:
#     for j in i:
#         fo_1.write(str(j)+" ")
# fo_1.close()

# fo_2 = open("temp_weights_h_out.txt", "w")
# for i in BPNN.weights_h_out:
#     for j in i:
#         fo_2.write(str(j)+" ")
# fo_2.close()

# fo_3 = open("temp_bias_in_h.txt", "w")
# # fo_3.write("bias_in_h:\n")
# for i in BPNN.bias_in_h:
#     fo_3.write(str(i)+" ")
# fo_3.close()

# fo_4 = open("temp_bias_h_out.txt", "w")
# # fo_4.write("bias_h_out")
# for i in BPNN.bias_h_out:
#     fo_4.write(str(i)+" ")
# fo_4.close()
