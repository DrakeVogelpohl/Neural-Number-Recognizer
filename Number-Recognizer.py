import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetlib import NeuralNet as NNet

# Import the training data
trainData = pd.read_csv('mnist_train.csv')
trainData = np.array(trainData)
trainSize, trainDim = trainData.shape
trainData = trainData.T
Y_train = trainData[0] # Desired Result
A0_train = trainData[1:trainDim]
A0_train = A0_train / 255.0 # Normalize values between 0 and 1

# Import the testing data
testData = pd.read_csv('mnist_test.csv')
testData = np.array(testData)
testSize, testDim = testData.shape
testData = testData.T
Y_test = testData[0] # Desired Result
A0_test = testData[1:testDim]
A0_test = A0_test / 255.0 # Normalize values between 0 and 1


# Neural Net Architecture
inputLayerSize = trainDim - 1 # The first col is the desired result so not part of the input size
outputLayerSize = 10
hiddenLayerSizes = [100, 10]

# Creating the network with random values
neuralNet = NNet(inputLayerSize, outputLayerSize, hiddenLayerSizes)


# Training Architecture
# Supported activation functions: "Sigmoid", "Tanh", "Softmax", "ReLU", "LeakyReLU", "ELU"
# Supported loss functions: "Mean Squared", "Cross Entropy"
# *Note: To use Cross entropy must use Softmax as the output Layer activation
iterations = 10000
stochasticGD = 1
alpha = 0.2
activationFunction = "LeakyReLU"
outActFunc = "Softmax"
lossFunc = "Cross Entropy"
print("Alpha: ", alpha)

# Training
print("\nTraining")
neuralNet.train(activationFunction, outActFunc, lossFunc, A0_train, Y_train, trainSize, iterations, alpha, 
                dispFreq=5000, SGD=stochasticGD, batchSize=100)


# Testing
print("\nTesting")
testOutput = neuralNet.test(activationFunction, outActFunc, A0_test, Y_test, testSize)

