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

dim_1_sel = [4,5,5,6,7,8]
dim_2_sel = [4,5,5,6,7,8]
num_mines_sel = [3,3,4,6,8,10]

fi_1 = open("weights_in_h.txt", "r")
line = fi_1.readline()
line = line.strip().split(' ')
for i in range(len(line)):
    line[i] = float(line[i])
line = np.array(line).reshape([input_size, hidden_size])
BPNN.weights_in_h = line
fi_1.close()

fi_2 = open("weights_h_out.txt", "r")
line = fi_2.readline()
line = line.strip().split(' ')
for i in range(len(line)):
    line[i] = float(line[i])
line = np.array(line).reshape([hidden_size, output_size])
BPNN.weights_h_out = line
fi_2.close()

fi_3 = open("bias_in_h.txt", "r")
line = fi_3.readline()
line = line.strip(' ').split(' ')
for i in range(len(line)):
    line[i] = float(line[i])
line = np.array(line)
BPNN.bias_in_h = line
fi_3.close()

fi_4 = open("bias_h_out.txt", "r")
line = fi_4.readline()
line = line.strip().split(' ')
for i in range(len(line)):
    line[i] = float(line[i])
line = np.array(line)
BPNN.bias_h_out = line
fi_4.close()

nums_succ = []

for b in range(6):
    num_succ_b = []
    for a in range(10):
        # test the model
        num_succ = 0
        for round in range(num_test_rounds):
            dim_1 = dim_1_sel[b]
            dim_2 = dim_2_sel[b]
            num_mines = num_mines_sel[b]
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
                prob /= count_prob
                sort_prob = []
                for i in range(dim_1):
                    for j in range(dim_2):
                        if(np.isnan(game.state[i][j])):
                            sort_prob.append((prob[i][j],[i,j]))
                sort_prob.sort()

                game.selectCell(sort_prob[0][1])
            if game.victory:
                num_succ += 1
        num_succ_b.append(num_succ)
    nums_succ.append(np.average(num_succ_b))
print(nums_succ)

fo = open("output.txt","w")
for num in nums_succ:
    fo.write(str(num)+" ")
fo.close()