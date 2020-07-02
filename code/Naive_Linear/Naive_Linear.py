import numpy as np

class Naive_Linear:
    def __init__(self, input, dim_1, dim_2, mine_num):
        # input
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.input = input
        # weights: w_0 to w_8
        # self.weights = np.ones(2+1)
        self.weights = np.random.rand(2+1)
        # output
        self.output = 0
        # average probability to be a mine
        self.average_prob = mine_num / (dim_1*dim_2)

    def padding(self,input):
        # padding around the field
        padding_array = np.ones([self.dim_1+2, self.dim_2+2])*self.average_prob*3
        padding_array[1:self.dim_1+1,1:self.dim_2+1] = input[0:self.dim_1,0:self.dim_2]
        return padding_array

    def travel(self):
        # output: probs
        probs = np.zeros([8,8])
        # padding input by 1
        self.input = self.padding(self.input)
        # 3x3 zone to handle
        zone = np.zeros((3,3))
        # travel through the field, with 3x3 zone
        for i in range(8):
            for j in range(8):
                zone[0:3,0:3] = self.input[i:i+3, j:j+3]
                zone[np.isnan(zone)] = self.average_prob * 8
                # probs[i,j] = np.sum(zone)
                probs[i,j] = self.calculate_prob(zone)
        return probs

    def calculate_prob(self, zone):
        # local weights
        weights = self.weights
        weights_sum = weights[0]+weights[1]
        weights[0] = weights[0] / weights_sum
        weights[1] = weights[1] / weights_sum

        # zone: 2 dimension -> x: 1 dimension
        x = np.zeros(9)
        for i in range(3):  x[i*3:i*3+3] = zone[i,...]
        
        # calculate prob
        prob = weights[0]*(x[0]+x[2]+x[6]+x[8]) + weights[1]*(x[1]+x[3]+x[5]+x[7]) + weights[2]*1
        return prob