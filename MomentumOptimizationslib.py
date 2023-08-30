import numpy as np

class MomentumOptimizations:
    def MomentumInitialization(momentum, w, b, numHiddenL):
        if momentum == "SGD with momentum":
            velocity_w = []
            velocity_b = []
            for i in range(numHiddenL + 1):
                velocity_w.append(np.zeros_like(w[i]))
                velocity_b.append(np.zeros_like(b[i]))
            return [velocity_w, velocity_b]
        elif momentum == "ADAM":
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

    def MomentumOptimizations(momentum):
        match momentum:
            case "SGD with momentum":
                return MomentumOptimizations.SGD_withMomentum
            case "ADAM":
                return MomentumOptimizations.ADAM
            case "No momentum":
                return MomentumOptimizations.noMomentum
            case default:
                raise Exception("Not an implemented loss function")
            

    def SGD_withMomentum(moments, dw, db, alpha, beta_1, beta_2, epsilon, t, numHiddenL):
        v_w = []
        v_b = []

        for i in range(numHiddenL + 1):
            # Calculating new velocities
            v_w.append(beta_1*moments[0][i] + alpha * dw[i])
            v_b.append(beta_1*moments[1][i] + alpha * db[i])
        return [v_w, v_b], [v_w, v_b]
    

    def ADAM(moments, dw, db, alpha, beta_1, beta_2, epsilon, t, numHiddenL):
        ## Use Kingma at al. (2015) ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION ##
        change_w = []
        change_b = []
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
            change_w.append(alpha / (np.sqrt(v_w[i]) + epsilon) * m_w[i])
            change_b.append(alpha / (np.sqrt(v_b[i]) + epsilon) * m_b[i])
        return [change_w, change_b], [m_w, m_b, v_w, v_b]
    

    def noMomentum(moments, dw, db, alpha, beta_1, beta_2, epsilon, t, numHiddenL):
        change_w = []
        change_b = []
        for i in range(numHiddenL + 1):
            change_w.append(alpha * dw[i])
            change_b.append(alpha * db[i])
        return [change_w, change_b], None