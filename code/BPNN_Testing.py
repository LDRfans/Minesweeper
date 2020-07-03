import numpy as np
import os
import random

from MineSweeper import MineSweeper
from BP_Neural_Network import Neural_Network

# set the attributes of BPNN
dim_1_BPNN = 4
dim_2_BPNN = 4
num_tiles_BPNN = dim_1_BPNN*dim_2_BPNN

# set the training and testing rounds
num_test_rounds = 1000

# calculate the number of input neurons = (# of tile) * (# of possiblities) + 1
input_size = dim_1_BPNN*dim_2_BPNN*10+1
# set the number of output neurons = # of tiles
output_size = dim_1_BPNN*dim_2_BPNN 
# calculate the number of hidden neurons = sqrt[(out+2)in] + 2sqrt[in/(out+2)])
hidden_size = int(np.sqrt((output_size+2)*input_size)+2*np.sqrt(input_size/(output_size+2)))
# create a BPNN  
BPNN = Neural_Network(input_size, hidden_size, output_size)

fo_1 = open("weights_in_h.txt", "r")
line = fo_1.readline()
line = line.strip().split(' ')
for i in range(len(line)):
    line[i] = float(line[i])
line = np.array(line).reshape([input_size, hidden_size])
BPNN.weights_in_h = line
fo_1.close()

fo_2 = open("weights_h_out.txt", "r")
line = fo_2.readline()
line = line.strip().split(' ')
for i in range(len(line)):
    line[i] = float(line[i])
line = np.array(line).reshape([hidden_size, output_size])
BPNN.weights_h_out = line
fo_2.close()

fo_3 = open("bias_in_h.txt", "r")
line = fo_3.readline()
line = line.strip(' ').split(' ')
for i in range(len(line)):
    line[i] = float(line[i])
line = np.array(line)
BPNN.bias_in_h = line
fo_3.close()

fo_4 = open("bias_h_out.txt", "r")
line = fo_4.readline()
line = line.strip().split(' ')
for i in range(len(line)):
    line[i] = float(line[i])
line = np.array(line)
BPNN.bias_h_out = line
fo_4.close()

# test the model
num_succ = 0
for round in range(num_test_rounds):
    dim_1 = 5
    dim_2 = 5
    num_mines = 3
    # create a Minesweeper game
    game = MineSweeper(dim_1, dim_2, num_mines)
    # start from the left-top tile
    game.selectCell([0,0])
    while not game.gameOver:
        min_to_next = np.zeros([1,3])
        prob = np.zeros([dim_1,dim_2])
        count_prob = np.zeros([dim_1,dim_2])

        for i in range(dim_1-dim_1_BPNN+1):
            for j in range(dim_2-dim_2_BPNN+1):
                BPNN.input = np.zeros(input_size)
                count = 0
                for ii in range(dim_1_BPNN):
                    for jj in range(dim_2_BPNN):
                        if np.isnan(game.state[i+ii,j+jj]):
                            BPNN.input[(ii*dim_2_BPNN+jj)*10+9] = 1.0
                            count += 1
                        else:
                            BPNN.input[int((ii*dim_2_BPNN+jj)*10+game.state[i+ii,j+jj])] = 1.0
                if count > 0:
                    BPNN.input[num_tiles_BPNN*10] = num_mines*num_tiles_BPNN/game.totalCells/count
                # let BPNN feeds forward
                BPNN.feedforward()

                temp = BPNN.output.copy().reshape([dim_1_BPNN,dim_2_BPNN])
                prob[i:i+dim_1_BPNN,j:j+dim_2_BPNN] += temp
                count_prob[i:i+dim_1_BPNN,j:j+dim_2_BPNN] += np.ones([dim_1_BPNN,dim_2_BPNN])
                # # add coordinate to output
                # output_with_coor = np.zeros([num_tiles_BPNN,2])
                # for ii in range(num_tiles_BPNN):
                #     output_with_coor[ii][0] = BPNN.output[ii]
                #     output_with_coor[ii][1] = ii
                
                # # find the minimum probability output
                # output_with_coor = output_with_coor[output_with_coor[:,0].argsort()]    
                # print(i,j)
                
                # print(np.round(temp.reshape([4,4]),decimals=2))
                # # check whether it is covered to break the loop
                # for ii in range(output_size):
                #     min_prob_out = output_with_coor[ii,1]
                #     next_tile_coor = [int(min_prob_out//dim_2_BPNN)+i,int(min_prob_out%dim_2_BPNN)+j]
                #     if np.isnan(game.state[next_tile_coor[0],next_tile_coor[1]]):
                #         min_to_next = np.vstack((min_to_next,[output_with_coor[ii,0],next_tile_coor[0],next_tile_coor[1]]))
                #         break
        prob /= count_prob
        sort_prob = []
        for i in range(dim_1):
            for j in range(dim_2):
                if(np.isnan(game.state[i][j])):
                    sort_prob.append((prob[i][j],[i,j]))
        sort_prob.sort()

        # min_to_next = min_to_next[1:]
        # min_to_next = min_to_next[min_to_next[:,0].argsort()]
        # min_out = min_to_next[0]
        # next_coor = [int(min_out[1]),int(min_out[2])]
        # print(next_coor)
        # print(min_to_next)
        game.selectCell(sort_prob[0][1])
        # print(game.state)
    #     print("----------------------------------------------------------------------")
    # print("-----------------------------------------------------------------------------------------")
    # print("-----------------------------------------------------------------------------------------")
    # check whether you have won
    if game.victory:
        num_succ += 1

print(num_succ)