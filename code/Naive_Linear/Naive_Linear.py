import numpy as np

class Naive_Linear:
    def __init__(self, state, neighbors, dim_1, dim_2, mine_num, weights, bias):
        # input
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.state = state
        self.input = state
        # weights: w_0 to w_8
        self.weights = weights
        # bias
        self.bias = bias
        # output
        self.output = 0
        # average probability to be a mine
        self.average_prob = mine_num / (dim_1*dim_2)

    def padding(self,input):
        # padding around the field
        padding_array = np.ones([self.dim_1+2, self.dim_2+2])*self.average_prob*8
        padding_array[1:self.dim_1+1,1:self.dim_2+1] = input[0:self.dim_1,0:self.dim_2]
        return padding_array

    def travel(self):
        # output: probs
        probs = np.zeros([self.dim_1,self.dim_2])
        # padding input by 1
        self.input = self.padding(self.input)
        # nan = average_prob * 8
        self.input[np.isnan(self.input)] = self.average_prob * 8
        # 3x3 zone to handle
        zone = np.zeros((3,3))
        # travel through the field, with 3x3 zone
        for i in range(self.dim_1):
            for j in range(self.dim_2):
                zone[0:3,0:3] = self.input[i:i+3, j:j+3]
                # probs[i,j] = np.sum(zone)
                probs[i,j] = self.calculate_prob(zone)
        return probs

    def calculate_prob(self, zone):
        # zone: 2 dimension -> x: 1 dimension
        x = np.zeros(9)
        for i in range(3):  x[i*3:i*3+3] = zone[i,...]

        # calculate prob
        prob = np.dot(self.weights,x)  + self.bias
        return prob

