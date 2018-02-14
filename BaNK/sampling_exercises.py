import numpy as np
import scipy as sc
from scipy.stats import multivariate_normal
from scipy.stats import invwishart, invgamma


def sampling_lambda_r_prior(meanData, varianceData, means, rprior):
    K = len(means)
    inver_varianceData = 1./varianceData
    mean = (meanData * inver_varianceData + rprior * np.sum(means))/(inver_varianceData + K * rprior)
    variance = 1./(inver_varianceData + K * rprior)
    lambdaprior = sc.stats.norm.rvs(loc=mean, scale=variance, size=1)

    shapeParameter = K + 1

    sum_net = 0
    for muj in means:
        sum_net += (muj - lambdaprior)**2

    scale = 1./(1./(K + 1) * (varianceData + sum_net))

    rprior = sc.stats.gamma.rvs(a=shapeParameter, scale=scale, size=1)

    return lambdaprior, rprior


def sampling_beta_omega_priors(betaPrior, variances, varianceData):
    K = len(variances)
    shapeParameter = K * betaPrior + 1
    precisionParameter = 1./variances
    scale = 1./(1. / (K * betaPrior + 1) * (1. / varianceData + betaPrior * np.sum(variances)))
    omegaPrior = sc.stats.gamma(a=shapeParameter, scale=scale, size=1)


    # Falta the adaptive rejection sample from Gilks & Wild 1992. para muestrear beta
    return omegaPrior, betaPrior

def sampling_means_variances(X, Z, variances, lambdaPrior, rprior, betaprior, omegaPrior):
    newMeans = []
    newVariances = []
    K = len(variances)
    Zk = np.array(Z)
    for k in range(K):
        # End samping the new mean
        Znk = np.where(Zk == k)
        nk = len(Znk)
        meank = np.mean(X[Znk])
        sk = 1./variances[k]
        mean = (meank*nk*sk + lambdaPrior*rprior)/(nk*sk + rprior)
        variance = 1./(nk*sk + rprior)
        muk = sc.stats.norm.rvs(loc=mean, scale=variance, size=1)
        newMeans.append(muk)
        # End samping the new mean
        # sampling Variance

        shapeParameter = betaprior + nk

        scaleParameter = 1./(1./(betaprior + nk) *(omegaPrior*betaprior + np.sum(i*i for i in X[Znk] - muk)))

        sk = sc.stats.gamma(a=shapeParameter, scale=scaleParameter, size=1)
        newVariances.append(1./sk**2)
        # End Samping Variancek

    return newMeans, newVariances


def sampling_Zj(alpha, Z, means, variances, X, lambdaprior, rprior, betaprior, omegaprior):
    Zk = np.array(Z)
    N = len(X)
    for i in range(N):
        p = []
        xi = X[i]
        K = len(means)
        numberOfClass = Z[i]
        for k in range(K):
            Znk = np.where(Zk == k)
            nk = len(Znk)
            if k in Znk:
                nk = nk - 1

            if nk > 0:
                sk = 1. / variances[numberOfClass]
                muk = means[numberOfClass]

            else: #new mean
                muk = sc.stats.norm.rvs(loc=lambdaprior, scale=1./rprior, size=1)
                sk = sc.stats.gamma.rvs(a=betaprior, scale=1./omegaprior, size=1)

            pactual = np.log(float(nk) / (N - 1 + alpha)) + 1. / 2 * np.log(sk) - sk * (xi - muk) ** 2 / 2
