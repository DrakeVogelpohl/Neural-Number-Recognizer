import numpy as np
## TODO
# implement momentum and different cost functions (Cross Entropy)
class NeuralNet:

    # Helper for init that creates weights and biases
    def __create_WB(self, inputLS, outputLS, hiddenLS, numHiddenL):
        w = [np.random.uniform(-1, 1, (hiddenLS[0], inputLS))]
        b = [np.random.uniform(-1, 1, (hiddenLS[0], 1))]

        for n in range(numHiddenL - 1):
            w.append(np.random.uniform(-1, 1, (hiddenLS[n+1], hiddenLS[n])))
            b.append(np.random.uniform(-1, 1, (hiddenLS[n+1], 1)))
            
        w.append(np.random.uniform(-1, 1, (outputLS, hiddenLS[numHiddenL - 1])))
        b.append(np.random.uniform(-1, 1, (outputLS, 1)))

        return w, b

    # Initialization
    def __init__(self, inLS, outLS, hLS):
        self.inputLS = inLS
        self.outputLS = outLS
        self.hiddenLS = hLS
        self.numHiddenL = len(self.hiddenLS)
        self.w, self.b = self.__create_WB(self.inputLS, self.outputLS, self.hiddenLS, self.numHiddenL)


    # Activation Functions and Derivatves
    def __Sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def __SigmoidDerivative(self, Z, da):
        sigDeriv = self.__Sigmoid(Z) * (1 - self.__Sigmoid(Z))
        return da * sigDeriv

    def __tanh(self, Z):
        return (2 * self.__Sigmoid(2*Z)) - 1

    def __tanhDerivative(self, Z, da):
        tanhDeriv = 1 - (self.__tanh(Z) * self.__tanh(Z))
        return da * tanhDeriv

    def __Softmax(self, Z):
        exp = np.exp(Z - Z.max())
        return exp / np.sum(exp, axis=0)
    
    def __SoftmaxDerivative(self, Z, da): 
        # I used The Maverick Meerkat's article as a guide #
        layerSize, _ = Z.shape
        sm = self.__Softmax(Z)
        sm = sm.T
        tensor1 = np.einsum('ij,ik->ijk', sm, sm) # (dataPoints, layerSize, layerSize)
        tensor2 = np.einsum('ij,jk->ijk', sm, np.eye(layerSize, layerSize))  # (dataPoints, layerSize, layerSize)
        dSM = tensor2 - tensor1
        dz = np.einsum('ijk,ik->ij', dSM, da.T)
        return dz.T

    def __ReLU(self, Z):
        return np.maximum(Z, 0)
    
    def __RelUDerivative(self, Z, da):
        ZCopy = Z.copy()
        ZCopy[Z<=0] = 0
        ZCopy[Z>0] = 1
        return da * ZCopy
    
    def __LeakyReLU(self, Z):
        negativeSlope = 0.05
        return np.maximum(Z, 0) + negativeSlope*np.minimum(0, Z)
    
    def __LeakyRelUDerivative(self, Z, da):
        negativeSlope = 0.05
        ZCopy = Z.copy()
        ZCopy[Z<=0] = negativeSlope
        ZCopy[Z>0] = 1
        return da * ZCopy
    
    def __ELU(self, Z):
        alpha = 0.5
        return np.maximum(Z, 0) + alpha*(np.exp(np.minimum(0, Z)) - 1)

    def __ELUDerivative(self, Z, da):
        alpha = 0.5
        ZCopy = Z.copy()
        ZCopy[Z>0] = 1
        eluDeriv = np.where(ZCopy > 0, ZCopy, alpha * np.exp(ZCopy))
        return da * eluDeriv



    # Foward Propagation
    def __forward_prop(self, A0, w, b, numHiddenL):
        a = [A0]
        z = []
        
        for n in range(numHiddenL):
            z.append(w[n].dot(a[n]) + b[n])
            a.append(self.__ELU(z[n]))
        z.append(w[numHiddenL].dot(a[numHiddenL]) + b[numHiddenL])
        a.append(self.__Softmax(z[numHiddenL]))
        return a, z


    # Back Propagation
    # The math was done on a separate doc linked in the ReadMe
    def __back_prop(self, y, a, z, w, numHiddenL, trainSize):
        # Last layer
        da = [2*(a[numHiddenL + 1] - y)]
        dz = [self.__SoftmaxDerivative(z[numHiddenL], da[0])]
        db = [((1/trainSize) * np.sum(dz[0], axis=1))]
        dw = [((1/trainSize) * dz[0].dot(a[numHiddenL].T))] 

        # Every other layer
        for n in range(numHiddenL):
            da.append(w[numHiddenL - n].T.dot(dz[n]))
            dz.append(self.__ELUDerivative(z[numHiddenL-1 - n], da[n+1]))
            db.append(((1/trainSize) * np.sum(dz[n+1], axis=1)))
            dw.append(((1/trainSize) * dz[n+1].dot(a[numHiddenL-1 -n].T)))    

        # Reverse order
        dw.reverse()
        db.reverse()
        return dw, db
    

    # Update weights and biases
    def __update_wb(self, w, b, dw, db, alpha, numHiddenL):
        new_w = []
        new_b = []
        for i in range(numHiddenL + 1):
            new_w.append(w[i] - alpha * dw[i])
            new_b.append(b[i] - np.reshape(alpha * db[i], (-1,1)))
        return new_w, new_b


    # Turns Y_train into usable data for our net. Does this by turing a number
    # into a vector the same size as the output layer with all indexes 0 except for
    # the one of the origional number
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



    # Public method that trains the net with the given data with the given number
    # of iterations and the learning rate alpha 
    def train(self, A0_train, Y_train, trainSize, iterations, alpha):
        y = self.__makeYUsable(Y_train , self.outputLS, trainSize)

        for i in range(iterations):
            a, z = self.__forward_prop(A0_train, self.w, self.b, self.numHiddenL)

            if (i+1) % 250 == 0 or i == 0:
                print("Iteration: ", (i+1))
                print("Accuracy: ", self.__get_accuracy(self.__get_predictions(a, self.numHiddenL), Y_train, trainSize))

            dw, db = self.__back_prop(y, a, z, self.w, self.numHiddenL, trainSize)
            self.w, self.b = self.__update_wb(self.w, self.b, dw, db, alpha, self.numHiddenL)
        return
    

    # Public method for testing the net with a given set of data. Prints the accuracy and returns
    # the final output layer
    def test(self, A0_test, Y_test, testSize):
        y = self.__makeYUsable(Y_test, self.outputLS, testSize)
        a, _ = self.__forward_prop(A0_test, self.w, self.b, self.numHiddenL)
        print("Accuracy: ", self.__get_accuracy(self.__get_predictions(a, self.numHiddenL), Y_test, testSize))
        return a[self.numHiddenL + 1]

