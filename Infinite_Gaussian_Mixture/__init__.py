from Develop_PhD_Student.Infinite_Gaussian_Mixture import infinite_GMM, printGMM, get_point_from_svg, learning_GMM
import numpy as np
import random


def generate_GMM(means, covariance, pik, size):
    X = []
    U = np.random.uniform(0,1,size)
    K = len(pik)
    K_vector = range(K)
    pikcum = np.cumsum(pik)
    for u in U:
        for k, p in zip(K_vector, pikcum):
            if p >= u:
                x = np.random.multivariate_normal(means[k], covariance[k], size=1)[0]
                X.append(x)
                break
    return np.array(X)


def __main():
    # mean1, cov1 = [0, 1], [[2, 0], [0, 3]]
    # mean2, cov2 = [0, 1], [[4, 0], [0, 2]]
    #
    # X1 = np.random.multivariate_normal(mean1, cov1, size=1000)
    # X2 = np.random.multivariate_normal(mean2, cov2, size=10000)
    #
    # X = np.vstack((X1, X2))
    #
    # class_learning = learning_GMM.learn_gmm(X,0.0005,10)
    # class_learning.kolmogorov_smirnoff_test(X, np.array(mean1), cov1)

    # means = [-80, 1.5, 9]
    # sigmak = [3, 4, 5]
    # number_of_samples = 300
    # X = np.random.normal(loc=means[0], scale=sigmak[0], size=number_of_samples*4)
    # X = np.append(X, np.random.normal(loc=means[1], scale=sigmak[1], size=number_of_samples))
    # #X = np.append(X, np.random.normal(loc=means[2], scale=sigmak[2], size=number_of_samples))
    # random.shuffle(X)
    # plotGMM = printGMM.plotgaussianmixture(X, np.array(means), np.array(sigmak), [4./5., 1/5.])
    # plotGMM.print_seperategaussians()
    # learning_Gmm = infinite_GMM.infinite_GMM(X, initial_number_class=5)
    # learning_Gmm.learn_GMM(200000)
    # plotGMM = printGMM.plotgaussianmixture(X, learning_Gmm.means, np.sqrt(1. / learning_Gmm.S), learning_Gmm.get_weights())
    # plotGMM.print_seperategaussians()



    means = np.array([[0, 1], [20, 60], [100, 100],[10,10]])
    cov = np.array([[[4, 0], [0, 2]], [[4, 0], [0, 2]], [[40, 0], [0, 2]],[[40, 0], [0, 2]] ])
    pik = np.array([1./4, 1./4, 1./4, 1./4])
    X = generate_GMM(means, cov, pik, size=900)
    plotGMM = printGMM.plotgaussianmixture(X, means, cov, pik)
    plotGMM.print_seperategaussians()

    random.shuffle(X)
    initialK = np.random.randint(2, 50)
    Z = np.random.randint(0, initialK, size=len(X))
    learning_Gmm = infinite_GMM.infinite_GMM(X, initial_number_class=initialK, initialZ=Z)
    learning_Gmm.learn_GMM(500)
    plotGMM = printGMM.plotgaussianmixture(X, learning_Gmm.means, learning_Gmm.get_covariance_matrix(),
                                           learning_Gmm.get_weights())
    plotGMM.print_seperategaussians()
    print("stop")
    # points = get_po
    # realmeans = np.array([np.mean(X1, axis=0), np.mean(X2, axis=0), np.mean(X3, axis=0), np.mean(X4, axis=0), np.mean(X5, axis=0)])
    # realCovariance = np.array([np.cov(X1.T), np.cov(X2.T), np.cov(X3.T), np.cov(X4.T), np.cov(X5.T)])
    #int_from_svg.get_points_by_svg("points_sample.svg")
    # X1 = np.array(points.get_group_by_id("g1"))
    # X2 = np.array(points.get_group_by_id("g2"))
    # X3 = np.array(points.get_group_by_id("g3"))
    # X4 = np.array(points.get_group_by_id("g4"))
    # X5 = np.array(points.get_group_by_id("g5"))
    #
    # realweights = np.array([len(X1), len(X2), len(X3), len(X4), len(X5)])
    # realweights = realweights/float(np.sum(realweights))
    # X = np.vstack((X1, X2, X3, X4, X5))
    # X_Gmm = generate_GMM(realmeans, realCovariance, realweights, 10000)
    # plotGMM = printGMM.plotgaussianmixture(X_Gmm, realmeans, realCovariance, realweights)
    #
    # plotGMM.print_seperategaussians()
    '''
    X = []
    with open("S1.txt") as f:
        for line in f:
            X.append(np.array(line.split("    ")[1:]).astype(dtype=float))

    X = np.array(X)

    learning_Gmm = infinite_GMM.infinite_GMM(X, initial_number_class=5)
    learning_Gmm.learn_GMM(5000)

    plotGMM = printGMM.plotgaussianmixture(X, learning_Gmm.means, learning_Gmm.get_covariance_matrix())
    plotGMM.printGMM()

    print("Means sampled: " + str(learning_Gmm.means))
    print("Sigma sampled: " + str(learning_Gmm.get_covariance_matrix()))
    '''
    '''

    
    '''



if __name__ == '__main__':
     __main()
