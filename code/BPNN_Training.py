import numpy as np
import os

from MineSweeper import MineSweeper
from BP_Neural_Network import Neural_Network

# set the attributes of the game
dim_1 = 8
dim_2 = 8
num_mines =10

# set the training and testing rounds
num_train_rounds = 100000
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

# train the model
for i in range(num_train_rounds):
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

        print(BPNN.weights_h_out)
        print("---------------------------------------------------------------------------")
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

# test the model
num_succ = 0
for i in range(num_test_rounds):
    # create a Minesweeper game
    game = MineSweeper(dim_1, dim_2, num_mines)
    # start from the left-top tile
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

    # check whether you have won
    if game.victory:
            num_succ += 1

fo_0 = open("output0.txt", "w")
fo_0.write(str(num_train_rounds)+" "+str(num_test_rounds)+" "+str(num_succ)+"\n")
fo_0.close()

fo_1 = open("output1.txt", "w")
fo_1.write("weights_in_h:\n")
for i in BPNN.weights_in_h:
    fo_1.write(str(i)+" ")
fo_1.close()

fo_2 = open("output2.txt", "w")
fo_2.write("weights_h_out:\n")
for i in BPNN.weights_h_out:
    fo_2.write(str(i)+" ")
fo_2.close()

fo_3 = open("output3.txt", "w")
fo_3.write("bias_in_h:\n")
for i in BPNN.bias_in_h:
    fo_3.write(str(i)+" ")
fo_3.close()

fo_4 = open("output4.txt", "w")
fo_4.write("bias_h_out")
for i in BPNN.bias_h_out:
    fo_4.write(str(i)+" ")

print("Done.")


