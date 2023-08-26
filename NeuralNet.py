import numpy as np

class NeuralNet:

    # Helper for init that creates weights and baises
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

    def __ReLU(Z):
        return np.max(Z, 0)


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
    def __back_prop(self, y, a, z, w, b):
        pass

    def train(self, A0, Y_train, itterations, alpha):
        pass