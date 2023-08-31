# Neural-Number-Recognizer
### Introduction
The machine learning behind this number recognizer is a vanilla neural network library built using only NumPy. Using this library to recognize numbers in the MNIST data set with greater than 98% testing accuracy.

### How it works
This library can be used to create custom architectures for vanilla neural networks. It works by calling init to initialization the network size and filling the initial weights and biases with variance distribution proposed by He et al. (2015). After initialization is training. Training is called by providing the training data and other classifiers/setting to train with. Then testing is callled and provides a final accuracy.

### Initialization
Initialization is called with 3 parameters. 
NNet(inputLayerSize, outputLayerSize, hiddenLayerSizes)
1. The input layer size
2. The output layer size
3. An array with the sizes of the desired hidden layers

For example, NNet(10, 5, [50, 10, 5]). Will result in a 10->50->10->5->5 foward prop network. 
### Training
Training supports industry standard activation and loss functions along with numerous optimizations.

Supported Stochastic Gradient Descent: SGD=1, batchSize='m'

Supported Momentums: momentum = "No momentum", "ADAM", "SGD with momentum"
    - ADAM: epsilon='e', beta_1='b1', beta_2='b2' 
        -default settings are: e=1e-8, beta_1=0.9, beta_2=0.999, alpha=0.001
    -For SGD with momentum: beta_1='b1' with default setting of beta_1=0.9
        -*Note: scale alpha by 0.1
 
Supported activation functions: "Sigmoid", "Tanh", "Softmax", "ReLU", "LeakyReLU", "ELU"

Supported loss functions: "Mean Squared", "Cross Entropy"
    -*Note: To use Cross entropy must use Softmax as the output Layer activation


train(self, actFunc, outpActFun, lossFunc, A0_train, Y_train, trainSize, iterations, alpha, 
              dispFreq=250, SGD=0, batchSize=100, momentum="No momentum", epsilon=1e-8, 
              beta_1=0.9, beta_2=0.999):
1. Activation function
2. Output (last) layer activation function 
3. Loss function 
3. The training input data with (n,m) shape. Where n is the number of classes and m is the number of examples
4. The training expected output with (n,m) shape.
5. The number of examples used in training or m as defined above
6. The number of itterations to train for.
7. The learning rate alpha

Non-Required parameters
1. How often to print the batch accuracy
2. If you want to use stochastic gradient descent: SGD=1 else use entire training data
3. Batch Size for SGD
4. If you want to use a supported momentum optimization
5. Epsilon value for ADAM
6. Beta value for SGD with momentum or Beta_1 value for ADAM
8. Beta_2 value for ADAM

The process for training is forward propagation, backpropagation, then updating the variables using gradient descent. Gradients are calculated during the backprop step.

The calculus and linear algebra for the backpropagation and gradient descent was done by hand. The calculations are linked here:
https://drive.google.com/file/d/16bgs6hfnzfMf96TNOH8ata-YxLZtJgjB/view?usp=sharing
### Testing
test(self, actFunc, outActFun, A0_test, Y_test, testSize)
Returns a vector of size m with the output predictions

Parameters:
1. Activation function
2. Output (last) layer activation function 
3. The testing input data with (n,m) shape. Where n is the number of classes and m is the number of examples
4. The testing expected output with (n,m) shape.  
5. The number of examples used in training or m as defined above 