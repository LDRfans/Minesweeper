import numpy as np

# function to calculate sigmiod function of input number x
def sigmoid(x):
    return 1/(1+np.exp(-x))

# function to calculate the derivative of sigmiod function w.r.t input number x
def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

class NeuralNetwork:
    def __init__(self, input, hidden_size, output_size, output_true_value):
        # input layer [input_layer, 1]
        self.input = input
        # weights between input layer and hidden layer
        self.weights_in_h = np.random.rand(self.input.shape[1],hidden_size)
        # weights between hidden layer and output layer 
        self.weights_h_out = np.random.rand(hidden_size ,output_size)
        # true value of output layer
        self.output_true_value = output_true_value
        # output layer
        self.output = np.zeros(output_size)

    def feedforward(self):
        self.hidden = sigmoid(np.dot(self.input, self.weights_in_h))
        self.output = sigmoid(np.dot(self.hidden, self.weights_h_out))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights_h_out = np.dot(self.hidden.T, (2*(self.output_true_value - self.output) * sigmoid_derivative(self.output)))
        d_weights_in_h = np.dot(self.input.T,  (np.dot(2*(self.output_true_value - self.output) * sigmoid_derivative(self.output), self.weights_h_out.T) * sigmoid_derivative(self.hidden)))

        # update the weights with the derivative (slope) of the loss function
        self.weights_in_h += d_weights_in_h
        self.weights_h_out += d_weights_h_out