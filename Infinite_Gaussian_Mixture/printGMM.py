import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.stats

class plotgaussianmixture:
    def __init__(self, X, means, covariances):
        self.X = X
        self.means = means
        self.covariances = covariances
        if len(self.covariances.shape) == 1:
            self.isOneDimension = True
            self.isTwoDimension = False
        elif self.covariances.shape[1] == 2:
            self.isTwoDimension = True
            self.isOneDimension = False
        else:
            self.isTwoDimension = False
            self.isOneDimension = False

    def __plotOneDimensionGMM(self):
        Y = np.zeros_like(self.X)
        plt.scatter(self.X, Y)

        for mean, sigma in zip(self.means, self.covariances):
            X = np.linspace(mean - 3*sigma, mean + 3*sigma,1000)

            Y = sc.stats.norm.pdf(X, loc=mean, scale=sigma)

            plt.plot(X, Y)

        plt.show()



    #def __plotTwoDimensionGMM(self):

    def printGMM(self):
        if self.isOneDimension:
            self.__plotOneDimensionGMM()
        #elif self.isTwoDimension:
            # self.__plotTwoDimensionGMM()
        else:
            print ("No es posible imprimir mas de dos dimensiones")