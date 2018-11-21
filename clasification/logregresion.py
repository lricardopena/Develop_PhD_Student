import numpy as np

class logreg:
    def __init__(self, X, Y, w0 = None, V0 = None, a0=0.0001, b0 = 0.0001):
        if w0 is None:
            w0 = np.zeros(len(X[0]))
        if V0 is None:
            V0 = 1./0.00001 * np.identity(len(X[0]))
            self.V0 = V0
            self.V0Inverse = 0.00001 * np.identity(len(X[0]))
        else:
            self.V0 = V0
            self.V0Inverse = np.linalg.inv(V0)

        self.VnInverse = self.V0Inverse + X.T.dot(X)
        self.Vn = np.linalg.inv(self.VnInverse)
        self.wn = self.Vn.dot(self.V0Inverse.dot(w0) + X.T.dot(Y))

        self.Y = Y


    def __sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def predict_Y(self, X, round=True):
        mu = X.dot(self.wn)
        if round:
            # return np.array(self.__sigmoid(sigma.dot(mu)), int)
            return np.array(np.round(self.__sigmoid(mu)), int)

        # return self.__sigmoid(sigma.dot(mu))
        return self.__sigmoid(mu)