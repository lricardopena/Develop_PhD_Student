'''
This implementation is made By: Luis Ricardo Pena Llamas, this code is an implementation of a Journal named:
Rasmussen, C. E. (2000). The infinite Gaussian mixture model. In Advances in neural information processing systems (pp. 554-560).
'''

import numpy as np
import scipy as sc
from scipy.stats import multivariate_normal


def samplingGMM(mus, sigmas, pik, N, D):
    X = []
    K = len(pik)
    u = np.random.uniform(size=N)
    for i in range(N):
        actualp = 0
        for k in range(K):
            actualp += pik[k]
            if actualp >= u[i]:
                if D == 1:
                    x = sc.stats.norm.rvs(loc=mus[k], scale=sigmas[k], size=1)
                else:
                    x = sc.stats.multivariate_normal.rvs(mean=mus[k], cov=sigmas[k], size=1)
                X.append(x)
                break

    return np.array(X)


def samplingLambda_r_univariate(means, meanx, sigmax, rprior):
    means = np.array(means)
    K = len(means)

    meanLambda = (meanx * 1. / sigmax + rprior * np.sum(means)) / (1. / sigmax + K * rprior)
    sigmaLambda = 1. / (1. / sigmax + K * rprior)
    lambdaPrior = sc.stats.norm.rvs(loc=meanLambda, scale=sigmaLambda, size=1)

    shapeRprior = K + 1
    scalePrior = 1. / (1. / (K + 1) * (sigmax) + np.sum((means - lambdaPrior) ** 2))

    rprior = np.random.gamma(shape=shapeRprior, scale=scalePrior, size=1)

    return lambdaPrior, rprior


def sampling_means_variances_univariate(X, Z, sigmas, lamdaPrior, rPrior, beta, w):
    K = len(sigmas)
    means = []
    newSigmas = []

    for k in range(K):
        Xk = X[np.where(Z == k)[0]]
        Nk = len(Xk)
        meanMuk = (np.mean(Xk) * Nk * 1. / sigmas[k] + lamdaPrior * rPrior) / (Nk * 1. / sigmas[k] + rPrior)
        varianceMuk = 1. / (Nk * 1. / sigmas[k] + rPrior)
        muk = sc.stats.norm.rvs(meanMuk, varianceMuk, size=1)[0]
        means.append(muk)

        shapeSigmak = beta + Nk
        scaleSigmak = 1. / (1. / (beta + Nk) * (w * beta + np.sum((Xk - muk) ** 2)))

        if scaleSigmak <= 0:
            # If the scale is near to cero whe cut
            scaleSigmak = 1. / 999999
        sigmak = 1. / np.random.gamma(shape=shapeSigmak, scale=scaleSigmak, size=1)[0]
        newSigmas.append(sigmak)

    return means, newSigmas


def sampling_w_beta_univariate(sigmas, beta, sigmax):
    K = len(sigmas)
    shapew = K * beta + 1
    scalew = 1. / (1. / (K * beta + 1) * (1. / sigmax + beta * np.sum(1. / np.array(sigmas))))

    w = np.random.gamma(shape=shapew, scale=scalew, size=1)

    '''
    Falta encontrar como muestrar beta utilizando Adaptive Rejection Sampling, formula (9.2) de "The inifinite Gaussian Mixture Model"
    '''

    return w, beta


def sampling_alpha_univariate(K, N):
    '''
    :param K: The number of clases
    :param N: The number of samples
    :return: alpha sampled
    '''

    '''
    Falta encontrar como muestrear alpha utilizando Adaptive Rejection Sampling, formula (15) de "The inifinite Gaussian Mixture Model"
    '''
    alpha = 1

    return alpha


def get_New_mu_sigma_univariate(lamdaPrior, rPrior, beta, w):
    muk = sc.random.normal(loc=lamdaPrior, scale=1. / rPrior, size=1)[0]
    sigmak = 1. / np.random.gamma(shape=beta, scale=1. / w, size=1)[0]

    return muk, sigmak


def samplingZ_univariate(X, Z, means, sigmas, alpha, lambdaPrior, rPrior, beta, w):
    N = len(X)
    K = len(means)
    orderVisit = range(N)
    np.random.shuffle(orderVisit)
    for i in orderVisit:
        logp = []

        for k in range(K):
            Nk = len(np.where(Z == k)[0])
            if Z[i] == k:
                Nk = Nk - 1

            if Nk > 0:
                p = np.log(Nk) - np.log(N - 1 + alpha)
            else:
                p = np.log(alpha) - np.log(N - 1 + alpha)
            p = p + sc.stats.norm.logpdf(x=X[i], loc=means[k], scale=sigmas[k])[0]
            logp.append(p)

        muk_new, sigmak_new = get_New_mu_sigma_univariate(lambdaPrior, rPrior, beta, w)
        p = np.log(alpha) - np.log(N - 1 + alpha) + sc.stats.norm.logpdf(x=X[i], loc=muk_new, scale=sigmak_new)[0]
        logp.append(p)
        '''
        Check the normalization step
        '''
        logp = (1. / np.array(logp) / np.sum(logp)) / np.sum(1. / np.array(logp) / np.sum(logp))

        actualp = 0
        u = np.random.uniform(size=1)
        for k in range(K + 1):
            actualp += logp[k]
            if actualp >= u:
                Z[i] = k
                if k >= K:
                    K += 1
                    means.append(muk_new)
                    sigmas.append(sigmak_new)
                break
        if len(np.unique(Z)) < len(means):
            K = len(means)
            for k in range(K):
                if len(np.where(Z == k)[0]) == 0:
                    K -= 1
                    del means[k]
                    del sigmas[k]
                    Z[np.where(Z > k)[0]] -= 1
                    break

    return Z, means, sigmas


def samplingUnivariateGMM(X, sweeps):
    N = len(X)
    mux = np.mean(X, axis=0)
    sigmax = np.std(X)
    means = []
    sigmas = []
    Z = np.array(list(np.random.randint(0, 12, size=N)))

    for k in range(len(np.unique(Z))):
        Xk = X[np.where(Z == k)[0]]
        muk = np.mean(Xk)
        sigmak = np.std(Xk)
        means.append(muk)
        sigmas.append(sigmak)

    # Priors
    rPrior = np.random.gamma(1, 1. / sigmax, size=1)[0]
    beta = 1. / np.random.gamma(1, 1, size=1)[0]
    w = np.random.gamma(1, sigmax, size=1)[0]
    alpha = 1. / np.random.gamma(1, 1, size=1)[0]
    for t in range(sweeps):
        lambdaPrior, rPrior = samplingLambda_r_univariate(means, mux, sigmax, rPrior)
        means, sigmas = sampling_means_variances_univariate(X, Z, sigmas, lambdaPrior, rPrior, beta, w)
        w, beta = sampling_w_beta_univariate(sigmas, beta, sigmax)
        Z, means, sigmas = samplingZ_univariate(X, Z, means, sigmas, alpha, lambdaPrior, rPrior, beta, w)
        K = len(means)
        alpha = sampling_alpha_univariate(K, N)

    return means, sigmas, Z


def main():
    realmean = np.array([20, 30, 100])
    realcov = np.array([2, 4, 6])
    realPik = np.array([1. / 2, 1. / 3, 1. / 4])
    realPik = realPik / np.sum(realPik)
    N = 1000
    X = samplingGMM(realmean, realcov, pik=realPik, N=N, D=1)
    means, covariances, Z = samplingUnivariateGMM(X, sweeps=1000)


if __name__ == "__main__":
    main()
