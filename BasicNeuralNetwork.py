# Author: Jared White
# Date Last Modified: 10/2/2019
# File Name: BasicNeuralNetwork.py
# File Description: Python script to demonstrate a basic neural net with a small data size. Employs the sigmoid activation function for propogation.
import numpy as np

# define neural network class
class Neural_Network(object):
    def __init__(self):
        # layer sizes
        self.inputSize = 2 # size of input data matrix
        self.outputSize = 1 # size of output data matrix
        self.hiddenSize = 3 # number of hidden layers

        # weights
        self.weight1 = np.random.randn(self.inputSize, self.hiddenSize) # weight matrix from input to hidden layers
        self.weight2 = np.random.randn(self.hiddenSize, self.outputSize) # weight matrix from hidden to output layers
    def forwardProp(self, input):
        self.h1 = np.dot(input, self.weight1) # dot product of the input data and the first weight matrix
        self.h2 = self.sigmoid(self.h1) # activation function calculations
        self.h3 = np.dot(self.h2, self.weight2) # dot product of the hidden layer and the second weight matrix
        return self.sigmoid(self.h3) # return the final activation function calculations
    def sigmoid(self, data):
        return 1/(1+np.exp((-1)*data)) # sigmoid activation function
    def sigmoid_prime(self, data):
        return data * (1 - data) # derivative of the sigmoid function
    def backProp(self, input_data, actual_output, predicted_output):
        # backward propogation through the network
        self.pred_error = actual_output - predicted_output # error in the output
        self.pred_delta = self.pred_error * self.sigmoid_prime(predicted_output) # apply sigmoid prime to error

        self.h2_error = self.pred_delta.dot(self.weight2.T) # error contributed by hidden layers
        self.h2_delta = self.h2_error*self.sigmoid_prime(self.h2) # apply sigmoid prime to hidden layers

        self.weight1 += input_data.T.dot(self.h2_delta) # recalculate weight 1 from backward propogation
        self.weight2 += self.h2.T.dot(self.pred_delta) # recalculate weight 2 from backward propogation
    def trainNetwork(self, input_data, output):
        pred_output = self.forwardProp(input_data) # forward propogation
        self.backProp(input_data, output, pred_output) # backward propogation
    def predictOutput(self, predicted_input):
        print("Predicted output based on trained weights: ")
        print("Weight 1: " + str(self.weight1))
        print("weight 2: " + str(self.weight2))
        print("Input (scaled): \n" + str(predicted_input))
        print("Output: \n" + str(self.forwardProp(predicted_input)))

# Setup training and testing data sets

# setup data models
input_data = np.array(([2, 5], [9, 4], [1, 8], [5, 10]), dtype=float) # input, some x1 and x2 is known to cause some result held in the output variable
output = np.array(([75], [86], [81]), dtype=float) # output

# scale units to be between 0 and 1
input_data = input_data/np.amax(input_data, axis=0)
output = output/100

# split models into training and testing sets
training_data = np.split(input_data, [3])[0]
testing_data = np.split(input_data, [3])[1]

# Train a neural net to predict results
neural_net = Neural_Network()

# Train the network 150000 times using the input and output data models
for i in range(0, 150000):
    print("Input: \n" + str(training_data))
    print("Actual Output: \n" + str(output))
    forward_prop = neural_net.forwardProp(training_data)
    print("Predicted Output: \n" + str(forward_prop))
    print("Loss: \n" + str(np.mean(np.square(output - forward_prop)))) # loss calculation
    print("\n")
    neural_net.trainNetwork(training_data, output)

# Output the predicted result
neural_net.predictOutput(testing_data)

