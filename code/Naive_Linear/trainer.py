import numpy as np
import random

from Naive_Play import naive_play
from Naive_Linear import Naive_Linear

def standard(x):
    x[4] = 0
    return x/sum(x)

def update(state, i, j, weights):
    zone = np.zeros([3,3])
    x = np.zeros(9)
    zone[0:3,0:3] = state[i:i+3,j:j+3]
    for t in range(3):  x[t*3:t*3+3] = zone[t,...]
    weights = weights + (1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x)))
    return weights


for n in range(1): # n models
    # init
    # solved squares num
    # weights = np.random.randn(9)
    weights = np.ones(9)
    weights[4] = 0
    weights = standard(weights)
    bias = 0

    for trainning_rounds in range(100000):
        result = naive_play(weights, bias)
        if result[3] == 1:
            wins += 1
        else:
            weights = standard(update(result[4], result[0], result[1], weights))
        if trainning_rounds%100 == 0:
            print(weights)
        
    # optimal
    opti_weights = weights
    opti_bias = bias
    opti_solved = 0
    wins = 0
    
    # testing
    testing_win = 0
    for testing in range(40):
        if naive_play(opti_weights, opti_bias)[3] == 1:    testing_win += 1
        
    print(opti_solved, opti_weights, opti_bias, "winning_rate:", testing_win/40)



# print(opti_weights)
# print(opti_bias)

