import infinite_GMM
import numpy as np
import printGMM

def __main():
    means = [-40, 0, 90]
    sigmak = [3, 4, 5]

    number_of_samples = 300
    X = np.random.normal(loc=means[0], scale=sigmak[0], size=number_of_samples)
    X = np.append(X, np.random.normal(loc=means[1], scale=sigmak[1], size=number_of_samples))
    X = np.append(X, np.random.normal(loc=means[2], scale=sigmak[2], size=number_of_samples))

    learning_Gmm = infinite_GMM.infinite_GMM(X, initial_number_class=5)
    learning_Gmm.learn_GMM(10000)

    plotGMM = printGMM.plotgaussianmixture(X, learning_Gmm.means, 1./learning_Gmm.S)

    plotGMM.printGMM()

    print "Means sampled: " + str(learning_Gmm.means)
    print "Real means: " + str(means)
    print "Sigma sampled: " + str(1./learning_Gmm.S)
    print "Real sigma: " + str(sigmak)


if __name__ == '__main__':
     __main()
