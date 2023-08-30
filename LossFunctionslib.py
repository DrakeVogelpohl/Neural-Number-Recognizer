import numpy as np

class LossFunctions:
    def lossFunctionSwitch(lossFunc):
        match lossFunc:
            case "Mean Squared":
                return LossFunctions.MeanSquaredError
            case "Cross Entropy":
                return LossFunctions.CrossEntropy
            case default:
                raise Exception("Not an implemented loss function")
            
    # Loss function derivatives
    def MeanSquaredError(a, y, z, outFunc):
        da = a - y
        return outFunc(z, da)
    
    def CrossEntropy(a, y, z, outFunc):
        grad = a - y
        _, batchSize = y.shape
        scaledGrad = grad / batchSize
        return grad