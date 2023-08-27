import numpy as np

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

    def __SigmoindDerivative(self, Z):
        return self.__Sigmoid(Z) * (1 - self.__Sigmoid(Z))
    
    def __tanh(self, Z):
        return (2 * self.__Sigmoid(2*Z)) - 1

    def __tanhDerivative(self, Z):
        return 1 - (self.__tanh(Z) * self.__tanh(Z))

    def __Softmax(self, Z):
        exp = np.exp(Z - Z.max())
        return exp / np.sum(exp)
    
    def __SoftmaxDerivative(self, Z, da, layerSize): 
        # I used The Maverick Meerkat's article as a guide #
        sm = self.__Softmax(Z)
        tensor1 = np.einsum('ik,jk->ijk', sm, sm) # (layerSize, layerSize, dataPoints)
        tensor2 = np.einsum('ik,ij->ijk', sm, np.eye(layerSize, layerSize))  # (layerSize, layerSize, dataPoints)
        dSM = tensor2 - tensor1
        return np.einsum('ijk,jk->ik', dSM, da)

    def __ReLU(self, Z):
        return np.maximum(Z, 0)
    
    def __RelUDerivative(self, Z):
        ZCopy = Z.copy()
        ZCopy[Z<=0] = 0
        ZCopy[Z>0] = 1
        return ZCopy
    
    def __LeakyReLU(self, Z):
        negativeSlope = 0.1
        return np.maximum(Z, 0) + negativeSlope*np.minimum(0, Z)
    
    def __LeakyRelUDerivative(self, Z):
        negativeSlope = 0.1
        ZCopy = Z.copy()
        ZCopy[Z<=0] = negativeSlope
        ZCopy[Z>0] = 1
        return Z
    
    def __ELU(self, Z):
        alpha = 0.1
        return np.maximum(Z, 0) + alpha(np.exp(np.minimum(0, Z)) - 1)

    def __ELUDerivative(self, Z):
        alpha = 0.1
        ZCopy = Z.copy()
        ZCopy[Z>0] = 1
        return np.where(ZCopy > 0, ZCopy, alpha * np.exp(ZCopy))



    # Foward Propagation
    def __forward_prop(self, A0, w, b, numHiddenL):
        a = [A0]
        z = []
        for n in range(numHiddenL):
            z.append(w[n].dot(a[n]) + b[n])
            a.append(self.__ReLU(z[n]))
        z.append(w[numHiddenL].dot(a[numHiddenL]) + b[numHiddenL])
        a.append(self.__Softmax(z[numHiddenL]))
        return a, z


    # Back Propagation
    # The math was done on a separate doc linked in the ReadMe
    def __back_prop(self, y, a, z, w, numHiddenL, trainSize):
        ## MIGHT HAVE TO NORMALIZE AT END ##
        # Need to check the math first + test both with the data and see which one is better

        # Last layer
        dz = [self.__SoftmaxDerivative(z[numHiddenL], 2*(a[numHiddenL + 1] - y), self.outputLS)]
        db = [((1/trainSize) * np.sum(dz, axis=1))]
        dw = [((1/trainSize) * np.sum(dz[0].dot(a[numHiddenL].T), axis=1))]

        # Every other layer
        for n in range(numHiddenL):
            dz.append(w[numHiddenL - n].T.dot(db[n]) * self.__RelUDerivative(z[numHiddenL-1 - n]))
            db.append(((1/trainSize) * np.sum(dz[n+1], axis=1)))
            dw.append(((1/trainSize) * np.sum(dz[n+1].dot(a[numHiddenL-1 -n].T), axis=1)))      

        # Reverse order
        dw.reverse()
        db.reverse()
        return dw, db
    

    # Update weights and biases
    def __update_wb(self, w, b, dw, db, alpha):
        new_w = w - dw * alpha
        new_b = b - db * alpha
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

            if (i+1) % 50 == 0 or i == 0:
                print("Iteration: ", (i+1))
                print("Accuracy: ", self.__get_accuracy(self.__get_predictions(a, self.numHiddenL), Y_train, trainSize))

            dw, db = self.__back_prop(y, a, z, self.w, self.numHiddenL, trainSize)
            self.w, self.b = self.__update_wb(self.w, self.b, dw, db, alpha)
        return
    

    # Public method for testing the net with a given set of data. Prints the accuracy and returns
    # the final output layer
    def test(self, A0_test, Y_test, testSize):
        y = self.__makeYUsable(Y_test, self.outputLS, testSize)
        a, _ = self.__forward_prop(A0_test, self.w, self.b, self.numHiddenL)
        print("Accuracy: ", self.get_accuracy(self.get_predictions(a, self.numHiddenL), Y_test, testSize))
        return a[self.numHiddenL + 1]

