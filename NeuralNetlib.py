import numpy as np
## TODO
# implement momentum
class NeuralNet:

    # Helper for init that creates weights and biases
    def __create_WB(self, inputLS, outputLS, hiddenLS, numHiddenL):
        ## Use He at al. (2015) variance to initialize weights ##
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


    # Activation Functions and Derivatves
    def __Sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def __SigmoidDerivative(self, Z, da):
        sigDeriv = self.__Sigmoid(Z) * (1 - self.__Sigmoid(Z))
        return da * sigDeriv

    def __Tanh(self, Z):
        return (2 * self.__Sigmoid(2*Z)) - 1

    def __TanhDerivative(self, Z, da):
        tanhDeriv = 1 - (self.__Tanh(Z) * self.__Tanh(Z))
        return da * tanhDeriv

    def __Softmax(self, Z):
        exp = np.exp(Z - np.max(Z, axis=0))
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
    
    def __ReLUDerivative(self, Z, da):
        ZCopy = Z.copy()
        ZCopy[Z<=0] = 0
        ZCopy[Z>0] = 1
        return da * ZCopy
    
    def __LeakyReLU(self, Z):
        negativeSlope = 0.05
        return np.maximum(Z, 0) + negativeSlope*np.minimum(0, Z)
    
    def __LeakyReLUDerivative(self, Z, da):
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

    def __activationSwitch(self, actFunc):
        match actFunc:
            case "Sigmoid":
                return self.__Sigmoid
            case "Tanh":
                return self.__Tanh
            case "Softmax":
                return self.__Softmax
            case "ReLU":
                return self.__ReLU
            case "LeakyReLU":
                return self.__LeakyReLU
            case "ELU":
                return self.__ELU
            case default:
                raise Exception("Not an implemented activation function")

    def ____activationDerivativeSwitch(self, actFunc):
        match actFunc:
            case "Sigmoid":
                return self.__SigmoidDerivative
            case "Tanh":
                return self.__TanhDerivative
            case "Softmax":
                return self.__SoftmaxDerivative
            case "ReLU":
                return self.__ReLUDerivative
            case "LeakyReLU":
                return self.__LeakyReLUDerivative
            case "ELU":
                return self.__ELUDerivative
            case default:
                raise Exception("Not an implemented activation function")


    # Loss function derivatives
    def __MeanSquaredError(self, a, y, z, outFunc):
        da = a - y
        return outFunc(z, da)
    
    def __CrossEntropy(self, a, y, z, outFunc):
        grad = a - y
        _, batchSize = y.shape
        scaledGrad = grad / batchSize
        return grad

    def __lossFunctionSwitch(self, lossFunc):
        match lossFunc:
            case "Mean Squared":
                return self.__MeanSquaredError
            case "Cross Entropy":
                return self.__CrossEntropy
            case default:
                raise Exception("Not an implemented loss function")



    # Foward Propagation
    def __forward_prop(self, actFunc, outpActFun, A0, w, b, numHiddenL):
        # Assigning the activation functions
        afunc = self.__activationSwitch(actFunc)
        ofunc = self.__activationSwitch(outpActFun)

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
        afunc = self.____activationDerivativeSwitch(actFunc)
        ofunc = self.____activationDerivativeSwitch(outpActFun)
        lfunc = self.__lossFunctionSwitch(lossFunc)

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
    def __update_wb(self, ADAM, epsilon, beta_1, beta_2, w, b, dw, db, alpha, moments, t, numHiddenL):
        ## Use Kingma at al. (2015) ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION ##

        new_w = []
        new_b = []
        if ADAM == 1:
            t += 1
            m_w = []
            m_b = []
            v_w = []
            v_b = []

            for i in range(numHiddenL + 1):
                # Calculating new first and second moment estimates
                m_w.append((beta_1*moments[0][i] + (1-beta_1)*dw[i]) / (1-beta_1**t))
                m_b.append((beta_1*moments[1][i] + (1-beta_1)*db[i]) / (1-beta_1**t))
                v_w.append((beta_2*moments[2][i] + (1-beta_2)* np.power(dw[i],2) ) / (1-beta_2**t))
                v_b.append((beta_2*moments[3][i] + (1-beta_2)* np.power(db[i],2) ) / (1-beta_2**t))

                # Calculating new weights and biases
                new_w.append(w[i] - (alpha / (np.sqrt(v_w[i]) + epsilon) * m_w[i]))
                new_b.append(b[i] - (alpha / (np.sqrt(v_b[i]) + epsilon) * m_b[i]))
            return new_w, new_b, [m_w, m_b, v_w, v_b]
        
        else:
            for i in range(numHiddenL + 1):
                new_w.append(w[i] - alpha * dw[i])
                new_b.append(b[i] - alpha * db[i])
            return new_w, new_b, None


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
    
    def __print_itterationAccuracy(self, dispFreq, itteration, SGD, Y_train, indexes, a, numHiddenL, batchSize):
        if (itteration+1) % dispFreq == 0 or itteration == 0:
            if SGD == 1:
                Y = Y_train[indexes]
            else:
                Y = Y_train
            print("Iteration: ", (itteration+1))
            print("Accuracy: ", self.__get_accuracy(self.__get_predictions(a, numHiddenL), Y, batchSize))
        return

    # Stochastic Gradient Descent
    def __SGD(self, SGD, A0, Y, trainSize, batchSize):
        if SGD == 1:
            indexes = np.random.choice(trainSize, batchSize)
            A0_new = A0[:,indexes]
            Y_new = Y[:,indexes]
            return A0_new, Y_new, batchSize, indexes
        else: 
            return A0, Y, trainSize, None
        
    def __ADAM(self, ADAM, w, b, numHiddenL):
        if ADAM == 1:
            m_w = []
            m_b = []
            v_w = []
            v_b = []
            for i in range(numHiddenL + 1):
                m_w.append(np.zeros_like(w[i]))
                m_b.append(np.zeros_like(b[i]))
                v_w.append(np.zeros_like(w[i]))
                v_b.append(np.zeros_like(b[i]))
            return [m_w, m_b, v_w, v_b]
        else:
            return None



    # Public method that trains the net with the given data with the given number
    # of iterations and the learning rate alpha 
    def train(self, actFunc, outpActFun, lossFunc, A0_train, Y_train, trainSize, iterations, alpha, 
              dispFreq=250, SGD=0, batchSize=100, ADAM=0, epsilon=1e-8, beta_1=0.9, beta_2=0.999):
        y = self.__makeYUsable(Y_train, self.outputLS, trainSize)
        moments = self.__ADAM(ADAM, self.w, self.b, self.numHiddenL)

        for i in range(iterations):
            A0_batch, y_batch, batchSize, indexes = self.__SGD(SGD, A0_train, y, trainSize, batchSize)

            a, z = self.__forward_prop(actFunc, outpActFun, A0_batch, self.w, self.b, self.numHiddenL)            
            dw, db = self.__back_prop(actFunc, outpActFun, lossFunc, y_batch, a, z, self.w, self.numHiddenL, batchSize)
            self.w, self.b, moments = self.__update_wb(ADAM, epsilon, beta_1, beta_2, self.w, self.b, dw, db, alpha, moments, i, self.numHiddenL)

            self.__print_itterationAccuracy(dispFreq, i, SGD, Y_train, indexes, a, self.numHiddenL, batchSize)
        return
    

    # Public method for testing the net with a given set of data. Prints the accuracy and returns
    # the final output layer
    def test(self, actFunc, outActFun, A0_test, Y_test, testSize):
        y = self.__makeYUsable(Y_test, self.outputLS, testSize)
        a, _ = self.__forward_prop(actFunc, outActFun, A0_test, self.w, self.b, self.numHiddenL)
        print("Accuracy: ", self.__get_accuracy(self.__get_predictions(a, self.numHiddenL), Y_test, testSize))
        return a[self.numHiddenL + 1]

