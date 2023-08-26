import numpy as np

class NeuralNet:

    # Helper for init that creates weights and biases
    def __create_WB(inputLS, outputLS, hiddenLS, numHiddenL):
        w = [2*np.random.rand(hiddenLS[0], inputLS) - 1]
        b = [2*np.random.rand(hiddenLS[0], 1) - 1]

        for n in range(numHiddenL - 1):
            w.append(2*np.random.rand(hiddenLS[n+1], hiddenLS[n]) - 1)
            b.append(2*np.random.rand(hiddenLS[n+1], 1) - 1)

        w.append(2*np.random.rand(outputLS, hiddenLS[numHiddenL - 1]) - 1)
        b.append(2*np.random.rand(outputLS, 1) - 1)
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
        return np.exp(Z) / np.sum(np.exp(Z))
    
    def __SoftmaxDerivative(Z):
        return Z

    def __ReLU(Z):
        return np.max(Z, 0)
    
    def __RelUDerivative(Z):
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
    def __back_prop(self, y, a, z, w, b, numHiddenL):
        dw = []
        db = [2*(a[numHiddenL + 1] - y) * self.__SoftmaxDerivative(z[numHiddenL])]

        for n in range(numHiddenL-1):
            db.append(w[numHiddenL - n].T.dot(db[n]) * self.__ReLUDerivative(z[numHiddenL-1 - n]))

        return dw.reverse(), db.reverse()


    # Turns Y_train into usable data for our net. Does this by turing a number
    # into a vector the same size as the output layer with all indexes 0 except for
    # the one of the origional number
    def __makeYUsable(self, Y, outputLS, trainSize):
        y = np.zeros(trainSize, outputLS)
        y[np.arange(trainSize), Y] = 1
        return y.T

    # Public method that trains the net with the given data with the given number
    # of iterations and the learning rate alpha 
    def train(self, A0, Y_train, iterations, alpha):
        pass

