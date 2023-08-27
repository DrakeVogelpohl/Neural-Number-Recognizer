import numpy as np

class NeuralNet:

    # Helper for init that creates weights and biases
    def __create_WB(inputLS, outputLS, hiddenLS, numHiddenL):
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
    def __Softmax(Z):
        exp = np.exp(Z - Z.max())
        return exp / np.sum(exp)
    
    def __SoftmaxDerivative(self, Z):
        sm = self.__Softmax(Z)
        jac = np.diagflat(Z) - np.dot(sm, sm.T)
        return Z

    def __ReLU(Z):
        return np.max(Z, 0)
    
    def __RelUDerivative(Z):
        Z[Z<=0] = 0
        Z[Z>0] = 1
        return Z


    # Foward Propagation
    def __forward_prop(self, A0, w, b, numHiddenL):
        a = [A0]
        z = []
        for n in range(numHiddenL):
            z.append(w[n].dot(a[n]) + b[n])
            a.append(self.__ReLU(z[n]))
        z.append(w[numHiddenL].dot(a[numHiddenL] + b[numHiddenL]))
        a.append(self.__Softmax(z[numHiddenL]))
        return a, z


    # Back Propagation
    # The math was done on a separate doc linked in the ReadMe
    def __back_prop(self, y, a, z, w, numHiddenL, trainSize):
        ## MIGHT HAVE TO NORMALIZE AT END ##
        # Need to check the math first + test both with the data and see which one is better

        # Last layer
        dCdbL = 2*(a[numHiddenL + 1] - y) * self.__SoftmaxDerivative(z[numHiddenL])
        db = [((1/trainSize) * np.sum(dCdbL, axis=1))]

        dCdwL = db[0].dot(a[numHiddenL].T)
        dw = [((1/trainSize) * np.sum(dCdwL, axis=1))]

        # Every other layer
        for n in range(numHiddenL):
            dCdb = w[numHiddenL - n].T.dot(db[n]) * self.__ReLUDerivative(z[numHiddenL-1 - n])
            db.append(((1/trainSize) * np.sum(dCdb, axis=1)))

            dCdw = db[n+1].dot(a[numHiddenL-1 -n].T)
            dw.append(((1/trainSize) * np.sum(dCdw, axis=1)))      

        # Reverse order
        dw.reverse()
        db.reverse()
        return dw, db
    

    # Update weights and biases
    def __update_wb(w, b, dw, db, alpha):
        new_w = w - dw * alpha
        new_b = b - db * alpha
        return new_w, new_b


    # Turns Y_train into usable data for our net. Does this by turing a number
    # into a vector the same size as the output layer with all indexes 0 except for
    # the one of the origional number
    def __makeYUsable(self, Y, outputLS, trainSize):
        y = np.zeros(trainSize, outputLS)
        y[np.arange(trainSize), Y] = 1
        return y.T
    
    # Get the predictions from the net
    def __get_predictions(a, numHiddenL):
        lastLayerOutput = a[numHiddenL + 1]
        return np.argmax(lastLayerOutput, 0)
    
    # Get the accuracy of the net compared to the data
    def __get_accuracy(predictions, Y, Y_size):
        return np.sum(predictions == Y) / Y_size



    # Public method that trains the net with the given data with the given number
    # of iterations and the learning rate alpha 
    def train(self, A0_train, Y_train, trainSize, iterations, alpha):
        y = self.__makeYUsable(Y_train)

        for i in range(iterations):
            a, z = self.__forward_prop(A0_train, self.w, self.b, self.numHiddenL)
            dw, db = self.__back_prop(y, a, z, self.w, self.numHiddenL, trainSize)
            self.w, self.b = self.__update_wb(self.w, self.b, dw, db, alpha)
            if (i+1) % 50 == 0:
                print("Iteration: ", (i+1))
                print("Accuracy: ", self.get_accuracy(self.get_predictions(a, self.numHiddenL), Y_train, trainSize))
        return
    

    # Public method for testing the net with a given set of data. Prints the accuracy and returns
    # the final output layer
    def test(self, A0_test, Y_test, testSize):
        y = self.__makeYUsable(Y_test)
        a, _ = self.__forward_prop(A0_test, self.w, self.b, self.numHiddenL)
        print("Accuracy: ", self.get_accuracy(self.get_predictions(a, self.numHiddenL), Y_test, testSize))
        return a[self.numHiddenL + 1]
