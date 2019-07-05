import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.stats
from matplotlib import patches


class plotgaussianmixture:
    def __init__(self, X, means, covariances, weights):
        self.X = X
        self.means = means
        self.covariances = covariances
        self.weights = weights
        if len(self.covariances.shape) == 1:
            self.isOneDimension = True
            self.isTwoDimension = False
        elif self.covariances.shape[1] == 2:
            self.isTwoDimension = True
            self.isOneDimension = False
        else:
            self.isTwoDimension = False
            self.isOneDimension = False

    def __plotOneDimensionGMM(self, plotpoints, separatedGaussians=False):
        starX = min(self.X - 3*max(self.covariances))
        stopX = max(self.X + 3*max(self.covariances))
        X = np.linspace(starX, stopX, 100000)
        Y = np.zeros_like(X)
        Y_points = np.zeros_like(self.X)

        for i in range(len(self.weights)):
            mean, sigma, pik = self.means[i], self.covariances[i], self.weights[i]
            if plotpoints:
                X = np.linspace(mean - 4*sigma, mean + 4*sigma, 10000)
                Y_points += pik*sc.stats.norm.pdf(self.X, loc=mean, scale=sigma)

            if separatedGaussians:
                X = np.linspace(mean - 4 * sigma, mean + 4*sigma, 10000)
                Y = pik * sc.stats.norm.pdf(X, loc=mean, scale=sigma)
                plt.plot(X, Y)
            else:
                Y += pik*sc.stats.norm.pdf(X, loc=mean, scale=sigma)

        if not separatedGaussians:
            plt.plot(X, Y)
        if plotpoints:
            plt.scatter(self.X, Y_points)
        plt.show()

    def __plotTwoDimensionGMM(self):
        fig, ax = plt.subplots()
        plt.scatter(self.X.T[0], self.X.T[1])
        for mean, sigma in zip(self.means, self.covariances):
            eigenvalues, eigenvectors = np.linalg.eig(sigma)
            index_max_eigenvalue = np.where(eigenvalues == max(eigenvalues))[0][0]
            index_min_eigenvalue = 1 - index_max_eigenvalue
            angle = np.arctan(eigenvectors[index_max_eigenvalue][1] / eigenvectors[index_max_eigenvalue][0])

            if angle < 0:
                angle += 2*np.pi

            e = patches.Ellipse(mean, 6*np.sqrt(eigenvalues[index_max_eigenvalue]), 6*np.sqrt(eigenvalues[index_min_eigenvalue]),
                            angle=angle, linewidth=2, fill=False, zorder=2)
            ax.add_patch(e)
        fig.tight_layout()
        plt.show()


    def printGMM(self, plotpoints=False):
        if self.isOneDimension:
            self.__plotOneDimensionGMM(plotpoints)
        elif self.isTwoDimension:
            self.__plotTwoDimensionGMM()
        else:
            print ("No es posible imprimir mas de dos dimensiones")

    def print_seperategaussians(self, plotpoints=False):
        if self.isOneDimension:
            self.__plotOneDimensionGMM(plotpoints, separatedGaussians=True)
        elif self.isTwoDimension:
            self.__plotTwoDimensionGMM()
        else:
            print ("No es posible imprimir mas de dos dimensiones")

    def __save_OneDimensionGMM(self, filename, separatedGaussians, svg_format):
        starX = min(self.X - 3*max(self.covariances))
        stopX = max(self.X + 3*max(self.covariances))
        X = np.linspace(starX, stopX, 100000)
        Y = np.zeros_like(X)
        Y_points = np.zeros_like(self.X)

        for i in range(len(self.weights)):
            mean, sigma, pik = self.means[i], self.covariances[i], self.weights[i]
            X = np.linspace(mean - 4*sigma, mean + 4*sigma, 10000)
            Y_points += pik*sc.stats.norm.pdf(self.X, loc=mean, scale=sigma)

            if separatedGaussians:
                X = np.linspace(mean - 4 * sigma, mean + 4*sigma, 10000)
                Y = pik * sc.stats.norm.pdf(X, loc=mean, scale=sigma)
                plt.plot(X, Y)
            else:
                Y += pik*sc.stats.norm.pdf(X, loc=mean, scale=sigma)

        if not separatedGaussians:
            plt.plot(X, Y)
        plt.scatter(self.X, Y_points)
        if svg_format:
            format = "svg"
        else:
            format = "png"

        plt.savefig(filename, format=format)

    def __save_TwoDimensionGMM(self, filename, svg_format):
        fig, ax = plt.subplots()
        plt.scatter(self.X.T[0], self.X.T[1])
        for mean, sigma in zip(self.means, self.covariances):
            eigenvalues, eigenvectors = np.linalg.eig(sigma)
            index_max_eigenvalue = np.where(eigenvalues == max(eigenvalues))[0][0]
            index_min_eigenvalue = 1 - index_max_eigenvalue
            angle = np.arctan(eigenvectors[index_max_eigenvalue][1] / eigenvectors[index_max_eigenvalue][0])

            if angle < 0:
                angle += 2*np.pi

            e = patches.Ellipse(mean, 6*np.sqrt(eigenvalues[index_max_eigenvalue]), 6*np.sqrt(eigenvalues[index_min_eigenvalue]),
                            angle=angle, linewidth=2, fill=False, zorder=2)
            ax.add_patch(e)
        fig.tight_layout()
        if svg_format:
            format = "svg"
        else:
            format = "png"
        plt.savefig(filename, format=format)

    def save_image(self, filename, separatedGaussians, svg_format=True):
        if self.isOneDimension:
            self.__save_OneDimensionGMM(filename, separatedGaussians, svg_format)
        elif self.isTwoDimension:
            self.__save_TwoDimensionGMM(filename, svg_format)
        else:
            print ("No es posible imprimir mas de dos dimensiones")