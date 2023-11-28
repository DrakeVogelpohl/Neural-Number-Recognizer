import numpy as np
from ActivationFunctionslib import ActivationFunctions as af
from LossFunctionslib import LossFunctions as lf
from MomentumOptimizationslib import MomentumOptimizations as mo

class NeuralNet:
# Training Architecture
    # Supported Stochastic Gradient Descent: SGD=1, batchSize='m'
    # Supported Momentums: momentum = "No momentum", "ADAM", "SGD with momentum"
        # For ADAM: epsilon='e', beta_1='b1', beta_2='b2' 
            # default settings are: e=1e-8, beta_1=0.9, beta_2=0.999, alpha=0.001
        # For SGD with momentum: beta_1='b1' with default setting of beta_1=0.9
            # *Note: scale alpha by 0.1
    # Supported activation functions: "Sigmoid", "Tanh", "Softmax", "ReLU", "LeakyReLU", "ELU"
    # Supported loss functions: "Mean Squared", "Cross Entropy"
        # *Note: To use Cross entropy must use Softmax as the output Layer activation

    # Helper for init that creates weights and biases
    def __create_WB(self, inputLS, outputLS, hiddenLS, numHiddenL):
        ## Use He et al. (2015) variance to initialize weights ##
        w_var = 2 / inputLS
        w = [np.random.normal(scale=w_var, size=(hiddenLS[0], inputLS))]
        b = [np.zeros((hiddenLS[0], 1))]

        for n in range(numHiddenL - 1):
            w_var = 2 / hiddenLS[n]
            w.append(np.random.normal(scale=w_var, size=(hiddenLS[n+1], hiddenLS[n])))
            b.append(np.zeros((hiddenLS[n+1], 1)))
            
        w_var = 2 / hiddenLS[numHiddenL - 1]
        w.append(np.random.normal(scale=w_var, size=(outputLS, hiddenLS[numHiddenL - 1])))
        b.append(np.zeros((outputLS, 1)))

        return w, b

    # Initialization
    def __init__(self, inLS, outLS, hLS):
        self.inputLS = inLS
        self.outputLS = outLS
        self.hiddenLS = hLS
        self.numHiddenL = len(self.hiddenLS)
        self.w, self.b = self.__create_WB(self.inputLS, self.outputLS, self.hiddenLS, self.numHiddenL)



    # Foward Propagation
    def __forward_prop(self, actFunc, outpActFun, A0, w, b, numHiddenL):
        # Assigning the activation functions
        afunc = af.activationSwitch(actFunc)
        ofunc = af.activationSwitch(outpActFun)

        a = [A0]
        z = []
        for n in range(numHiddenL):
            z.append(w[n].dot(a[n]) + b[n])
            a.append(afunc(z[n]))
        z.append(w[numHiddenL].dot(a[numHiddenL]) + b[numHiddenL])
        a.append(ofunc(z[numHiddenL]))
        return a, z


    # Back Propagation
    # The math was done on a separate doc linked in the ReadMe
    def __back_prop(self, actFunc, outpActFun, lossFunc, y, a, z, w, numHiddenL, trainSize):
        # Assigning the activation and loss functions
        afunc = af.activationDerivativeSwitch(actFunc)
        ofunc = af.activationDerivativeSwitch(outpActFun)
        lfunc = lf.lossFunctionSwitch(lossFunc)

        # Last layer
        da = []
        dz = [lfunc(a[numHiddenL + 1], y, z[numHiddenL], ofunc)]
        db = [np.reshape(((1/trainSize) * np.sum(dz[0], axis=1)), (-1,1))]
        dw = [((1/trainSize) * dz[0].dot(a[numHiddenL].T))] 

        # Every other layer
        for n in range(numHiddenL):
            da.append(w[numHiddenL - n].T.dot(dz[n]))
            dz.append(afunc(z[numHiddenL-1 - n], da[n]))
            db.append(np.reshape(((1/trainSize) * np.sum(dz[n+1], axis=1)), (-1,1)))
            dw.append(((1/trainSize) * dz[n+1].dot(a[numHiddenL-1 -n].T)))    

        # Reverse order
        dw.reverse()
        db.reverse()
        return dw, db
    

    # Update weights and biases
    def __update_wb(self, momentum, epsilon, beta_1, beta_2, w, b, dw, db, alpha, moments, t, numHiddenL):
        momentumFunc = mo.MomentumOptimizations(momentum)

        new_w = []
        new_b = []
        change, moments = momentumFunc(moments, dw, db, alpha, beta_1, beta_2, epsilon, t, numHiddenL)
        for i in range(numHiddenL + 1):
            new_w.append(w[i] - change[0][i])
            new_b.append(b[i] - change[1][i])
        return new_w, new_b, moments



    # Stochastic Gradient Descent
    def __SGD(self, SGD, A0, Y, trainSize, batchSize):
        if SGD == 1:
            indexes = np.random.choice(trainSize, batchSize)
            A0_new = A0[:,indexes]
            Y_new = Y[:,indexes]
            return A0_new, Y_new, batchSize, indexes
        else: 
            return A0, Y, trainSize, None
        

    # Turns Y_train into usable data for the net through one hot encoding. 
    # Does this by turing a number into a vector the same size as the output 
    # layer with all indexes 0 except for the one of the origional number.
    def __makeYUsable(self, Y, outputLS, trainSize):
        y = np.zeros((trainSize, outputLS))
        y[np.arange(trainSize), Y] = 1
        return y.T
    
    # Get the predictions from the net
    def __get_predictions(self, a, numHiddenL):
        lastLayerOutput = a[numHiddenL + 1]
        return np.argmax(lastLayerOutput, 0)
    
    # Get the accuracy of the net compared to the data
    def __get_accuracy(self, predictions, Y, Y_size):
        return np.sum(predictions == Y) / Y_size
    
    # Print the accuracy during training at the given frequency 
    def __print_itterationAccuracy(self, dispFreq, itteration, SGD, Y_train, indexes, a, numHiddenL, batchSize):
        if (itteration+1) % dispFreq == 0 or itteration == 0:
            if SGD == 1:
                Y = Y_train[indexes]
            else:
                Y = Y_train
            print("Iteration: ", (itteration+1))
            print("Accuracy: ", self.__get_accuracy(self.__get_predictions(a, numHiddenL), Y, batchSize))
        return



    # Public method that trains the net with the given data with the given number
    # of iterations and the learning rate alpha 
    def train(self, actFunc, outpActFun, lossFunc, A0_train, Y_train, trainSize, iterations, alpha, 
              dispFreq=250, SGD=0, batchSize=100, momentum="No momentum", epsilon=1e-8, 
              beta_1=0.9, beta_2=0.999):
        
        y = self.__makeYUsable(Y_train, self.outputLS, trainSize)
        moments = mo.MomentumInitialization(momentum, self.w, self.b, self.numHiddenL)

        for i in range(iterations):
            A0_batch, y_batch, batchSize, indexes = self.__SGD(SGD, A0_train, y, trainSize, batchSize)

            a, z = self.__forward_prop(actFunc, outpActFun, A0_batch, self.w, self.b, self.numHiddenL)            
            dw, db = self.__back_prop(actFunc, outpActFun, lossFunc, y_batch, a, z, self.w, self.numHiddenL, batchSize)
            self.w, self.b, moments = self.__update_wb(momentum, epsilon, beta_1, beta_2, self.w, self.b, dw, db, alpha, moments, i, self.numHiddenL)

            self.__print_itterationAccuracy(dispFreq, i, SGD, Y_train, indexes, a, self.numHiddenL, batchSize)
        return
    

    # Public method for testing the net with a given set of data. Prints the accuracy and returns
    # the final output layer
    def test(self, actFunc, outActFun, A0_test, Y_test, testSize):
        a, _ = self.__forward_prop(actFunc, outActFun, A0_test, self.w, self.b, self.numHiddenL)
        print("Accuracy: ", self.__get_accuracy(self.__get_predictions(a, self.numHiddenL), Y_test, testSize))
        return self.__get_predictions(a, self.numHiddenL)

