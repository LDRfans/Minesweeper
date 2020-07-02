import numpy as np

class Naive_Linear:
    def __init__(self, input, dim_1, dim_2, mine_num):
        # input
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.input = input
        # weights: w_0 to w_8
        self.weights = np.random.rand(8+1)
        # output
        self.output = 0
        # average probability to be a mine
        self.average_prob = mine_num / (dim_1*dim_2)

    def padding(self,input):
        padding_array = np.zeros([self.dim_1+2, self.dim_2+2])
        # for i in range(self.dim_1):
        #     for j in range(self.dim_2):
        padding_array[1:self.dim_1+1,1:self.dim_2+1] = input[0:self.dim_1,0:self.dim_2]
        return padding_array

    def sweep(self):
        # padding input by 1
        self.input = self.padding(self.input)
        # 3x3 zone to handle
        zone = np.zeros((3,3))
        # 
        for i in range(8):
            for j in range(8):
                zone[0:3,0:3] = self.input[i:i+3, j:j+3]
                zone[np.isnan(zone)] = self.average_prob
            w = np.sum(zone)

        return w
