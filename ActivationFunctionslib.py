import numpy as np

class ActivationFunctions:
    def activationSwitch(actFunc):
        match actFunc:
            case "Sigmoid":
                return ActivationFunctions.Sigmoid
            case "Tanh":
                return ActivationFunctions.Tanh
            case "Softmax":
                return ActivationFunctions.Softmax
            case "ReLU":
                return ActivationFunctions.ReLU
            case "LeakyReLU":
                return ActivationFunctions.LeakyReLU
            case "ELU":
                return ActivationFunctions.ELU
            case default:
                raise Exception("Not an implemented activation function")

    def activationDerivativeSwitch(actFunc):
        match actFunc:
            case "Sigmoid":
                return ActivationFunctions.SigmoidDerivative
            case "Tanh":
                return ActivationFunctions.TanhDerivative
            case "Softmax":
                return ActivationFunctions.SoftmaxDerivative
            case "ReLU":
                return ActivationFunctions.ReLUDerivative
            case "LeakyReLU":
                return ActivationFunctions.LeakyReLUDerivative
            case "ELU":
                return ActivationFunctions.ELUDerivative
            case default:
                raise Exception("Not an implemented activation function")
    
    # Activation Functions and Derivatves
    def Sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    def SigmoidDerivative(Z, da):
        sigDeriv = ActivationFunctions.Sigmoid(Z) * (1 - ActivationFunctions.Sigmoid(Z))
        return da * sigDeriv

    def Tanh(Z):
        return (2 * ActivationFunctions.Sigmoid(2*Z)) - 1

    def TanhDerivative(Z, da):
        tanhDeriv = 1 - (ActivationFunctions.Tanh(Z) * ActivationFunctions.Tanh(Z))
        return da * tanhDeriv

    def Softmax(Z):
        exp = np.exp(Z - np.max(Z, axis=0))
        return exp / np.sum(exp, axis=0)
    
    def SoftmaxDerivative(Z, da): 
        # I used The Maverick Meerkat's article as a guide #
        layerSize, _ = Z.shape
        sm = ActivationFunctions.Softmax(Z)
        sm = sm.T
        tensor1 = np.einsum('ij,ik->ijk', sm, sm) # (dataPoints, layerSize, layerSize)
        tensor2 = np.einsum('ij,jk->ijk', sm, np.eye(layerSize, layerSize))  # (dataPoints, layerSize, layerSize)
        dSM = tensor2 - tensor1
        dz = np.einsum('ijk,ik->ij', dSM, da.T)
        return dz.T

    def ReLU(Z):
        return np.maximum(Z, 0)
    
    def ReLUDerivative(Z, da):
        ZCopy = Z.copy()
        ZCopy[Z<=0] = 0
        ZCopy[Z>0] = 1
        return da * ZCopy
    
    def LeakyReLU(Z):
        negativeSlope = 0.05
        return np.maximum(Z, 0) + negativeSlope*np.minimum(0, Z)
    
    def LeakyReLUDerivative(Z, da):
        negativeSlope = 0.05
        ZCopy = Z.copy()
        ZCopy[Z<=0] = negativeSlope
        ZCopy[Z>0] = 1
        return da * ZCopy
    
    def ELU(Z):
        alpha = 0.5
        return np.maximum(Z, 0) + alpha*(np.exp(np.minimum(0, Z)) - 1)

    def ELUDerivative(Z, da):
        alpha = 0.5
        ZCopy = Z.copy()
        ZCopy[Z>0] = 1
        eluDeriv = np.where(ZCopy > 0, ZCopy, alpha * np.exp(ZCopy))
        return da * eluDeriv