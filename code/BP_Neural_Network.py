import numpy as np

# function to calculate sigmiod function of input number x
def sigmoid(x):
    return 1/(1+np.exp(-x))

# function to calculate the derivative of sigmiod function w.r.t input number x
def sigmoid_derivative(x):
    return x*(1-x)

class Neural_Network:
    def __init__(self, input_size, hidden_size, output_size):
        # input layer [input_layer, 1]
        self.input = np.zeros(input_size)
        # weights between input layer and hidden layer
        self.weights_in_h = np.random.normal(0,4*np.sqrt(2/(input_size+hidden_size)),[input_size, hidden_size])
        # bias between input layer and hidden layer
        self.bias_in_h = np.zeros(hidden_size)
        # weights between hidden layer and output layer 
        self.weights_h_out = np.random.normal(0,4*np.sqrt(2/(hidden_size+output_size)),[hidden_size, output_size])
        # bias between hidden layer and output layer
        self.bias_h_out = np.zeros(output_size)        
        # output layer
        self.output = np.zeros(output_size)
        # true value of output
        self.output_true_value = np.zeros(output_size)

    def feedforward(self):
        self.hidden = sigmoid(np.dot(self.input, self.weights_in_h)+self.bias_in_h)
        self.output = sigmoid(np.dot(self.hidden, self.weights_h_out)+self.bias_h_out)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights_h_out = np.dot(np.mat(self.hidden).T, np.mat((2*(self.output_true_value - self.output) * sigmoid_derivative(self.output))))
        d_weights_in_h = np.dot(np.mat(self.input).T, np.mat(np.dot(2*(self.output_true_value - self.output) * sigmoid_derivative(self.output), self.weights_h_out.T) * sigmoid_derivative(self.hidden)))
        d_bias_h_out = 2*(self.output_true_value - self.output) * sigmoid_derivative(self.output)
        d_bias_in_h = np.dot(2*(self.output_true_value - self.output) * sigmoid_derivative(self.output), self.weights_h_out.T) * sigmoid_derivative(self.hidden)

        # update the weights with the derivative (slope) of the loss function
        # temp = self.weights_h_out.copy()
        self.weights_in_h += d_weights_in_h
        self.weights_h_out += d_weights_h_out
        self.bias_in_h += d_bias_in_h
        self.bias_h_out += d_bias_h_out

        # compare0 = (self.weights_h_out == temp).all()
        # print(compare0)
        # differece = self.weights_h_out - temp
        # compare1 = (differece == d_weights_h_out).all()
        # print(compare1)
        # print("----------------------------------------------------------------")