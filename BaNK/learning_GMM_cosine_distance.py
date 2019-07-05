import numpy as np
from scipy.stats import beta
from scipy.stats import multivariate_normal
import scipy as sc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import math
import sys


# Simultaed mixted weghts using stick break GEM, move type a
def updateMixingWeights(delta, Z):
    k = len(Z)
    q = np.zeros(k)
    u = np.zeros(k)
    nk = []
    for zk in Z:
        nk.append(len(zk))
    nk = np.array(nk)
    alphaVect = np.tile(delta, k) + nk
    for j in range(0, k-1):
        u[j] = beta.rvs(alphaVect[j], np.sum(alphaVect[j+1:]))
        q[j] = u[j]
        if j > 0:
            for i in range(0,j):
                q[j] = q[j]* (1 - u[i])
    q[k-1] = 1 - np.sum(q)
    return q
    #maybe this is not use because is a gibbs sampling


def InitializeZ(pi, means, sigmas, X):
    N = len(X)
    M = len(means)
    Zgroups = []
    u = np.random.uniform(size = N)
    sigmaInverse = []
    constants = []
    for m in range(0, M):
        constants.append(np.log(pi[m]) + np.log(np.power(np.linalg.det(sigmas[m]), -1.0/2)))
        sigmaInverse.append(np.linalg.inv(sigmas[m]))
        Zgroups.append([])

    constants = np.array(constants)
    for n in range(0, N):
        p = []
        for s in range(0, M):
            p.append( -1*(constants[s] -1.0/2 * np.dot(np.dot((X[n] - means[s]).T, sigmaInverse[s]),(X[n] - means[s]))))

        p = np.array(p)
        p = np.sum(p) / p
        p = p / np.sum(p)
        prob = 0
        s = -1
        for pactual in p:
            prob += pactual
            s += 1
            if prob >= u[n]:
                Zgroups[s].append(n)
                break

    return Zgroups


def E_step(X, means, sigmas, pik):
    K = len(means)
    W = []

    for x in X:
        w = []
        for k in range(K):
            wik = np.log(pik[k]) + sc.stats.multivariate_normal.logpdf(x, means[k], sigmas[k])
            w.append(wik)
        w = w / np.sum(w)
        W.append(w)

    return np.array(W)

def M_step(W, X, K):
    means, sigmas, pik = [], [], []
    N = len(X)
    d = X.shape[1]

    for k in range(K):
        Nk = np.sum(W.T[k])

        # Computing muk_new
        muk = np.zeros_like(X[0])
        for n in range(N):
            muk += W[n][k] * X[n]

        muk = 1./Nk * muk

        means.append(muk)

        # Computing sigmak_new
        sigmak = np.zeros((d, d))
        for n in range(N):
            rest = X[n] - muk
            rest = rest.reshape(-1, rest.shape[0])
            sigmak += W[n][k] * rest.T.dot(rest)

        sigmak = 1./Nk * sigmak

        sigmas.append(sigmak)

        # Computing pik
        pi = float(Nk)/N

        pik.append(pi)

    return np.array(means), np.array(sigmas), np.array(pik)

def samplingGMM(N, means, cov, pi):
    sampling = np.zeros((N, means.shape[1]))
    # sampling = np.zeros((N, 1)) #1-Dimension
    for i in xrange(N):
        randomNumber = np.random.uniform()
        current = 0
        index = 0
        for actualpi in pi:
            current += actualpi
            if current > randomNumber:
                xi = multivariate_normal.rvs(mean=means[index], cov=cov[index], size=1)
                sampling[i] = xi
                break
            index += 1

    return sampling

def GEM(alpha, k):
    q = np.zeros(k)
    u = np.zeros(k)
    alphaVect = np.tile(alpha, k)
    for j in range(0, k - 1):
        u[j] = beta.rvs(alphaVect[j], np.sum(alphaVect[j + 1:]))
        q[j] = u[j]
        if j > 0:
            for i in range(0, j):
                q[j] = q[j] * (1 - u[i])
    q[k - 1] = 1 - np.sum(q)
    return q


def updateAlocation(pik, means, sigmas, X, Z):
    N = len(X)
    K = len(means)
    Z = list(Z)
    Zgroups = []
    u = np.random.uniform(size = N)
    for k in range(K):
        Zgroups.append([])

    for n in range(0, N):
        p = []
        for s in range(0, K):
            p.append(sc.stats.multivariate_normal.logpdf(X[n],means[s], sigmas[s]))
            # p.append( -( constants[s] + (-1.0/2 * np.dot(np.dot((X[n] - means_omega[s]).T, sigmaInverse[s]), (X[n] - means_omega[s])))) )

        if not np.sum(p) == 0:
            p = np.array(p)
            p = np.sum(p) / p
            p = p / np.sum(p)
        else:
            p = np.tile([1./ K], K)

        probabilityOfAceptance = 0
        s = -1

        for pactual in p:
            probabilityOfAceptance += pactual
            s += 1
            if u[n] <= probabilityOfAceptance:
                Zgroups[s].append(n)
                break


    #its a gibbs sampling, so always accept
    return Zgroups
    oldLikelihood = logLikelihood(Y=X, means=means, sigmas=sigmas, Z=Z, pik=pi)
    newLikelihood = logLikelihood(Y=X, means=means, sigmas=sigmas, Z=Zgroups, pik=pi)

    probabilityOfAceptance = np.array((1, np.exp(newLikelihood - oldLikelihood))).min()
    u = np.random.uniform(size=1)[0]
    if u <= probabilityOfAceptance:
        return Zgroups
    return Z


#Return the log likelihood
def logLikelihood(Y, means, sigmas, pik):
    resultLoglikelihood = 0

    K = len(means)

    for x in X:
        for k in range(K):
            resultLoglikelihood += np.log(pik[k]) + sc.stats.multivariate_normal.logpdf(x, means[k], sigmas[k])


    return resultLoglikelihood



def sortParameters(means, indexUnsorted):
    if len(indexUnsorted) <= 1:
        return indexUnsorted

    half = int(math.floor(len(indexUnsorted)/2))
    L = indexUnsorted[ :  half]
    R = indexUnsorted[half:]
    leftIndexSorted = sortParameters(means, L)
    rightIndexSorted = sortParameters(means, R)

    indexSorted = []
    while len(leftIndexSorted) > 0 or len(rightIndexSorted) > 0:
        lIndex = len(leftIndexSorted) - 1
        rIndex = len(rightIndexSorted) - 1
        if lIndex >= 0 and rIndex >= 0:
            lValue, rValue = np.linalg.norm(means[leftIndexSorted[0]]), np.linalg.norm(means[rightIndexSorted[0]])
        elif lIndex == -1:
            lValue, rValue = sys.float_info.max,  np.linalg.norm(means[rightIndexSorted[0]])
        else:
            lValue, rValue = np.linalg.norm(means[leftIndexSorted[0]]), sys.float_info.max

        if lValue < rValue:
            indexSorted.append(leftIndexSorted[0])
            del leftIndexSorted[0]
        else:
            indexSorted.append(rightIndexSorted[0])
            del rightIndexSorted[0]


    return indexSorted

def adjustIndex(pi, means, lambdaInverse, xi, tau, ellInverse, r, Z):
    sortedIndex = sortParameters(means, range(0, len(means)))
    return pi[sortedIndex], means[sortedIndex], lambdaInverse[sortedIndex], xi[sortedIndex], tau[sortedIndex], ellInverse[sortedIndex], r[sortedIndex], list(np.array(Z)[sortedIndex])


N = 1500
#realmeans, realcov, realpik = np.array([[175, 55, 120], [125,110,1], [30,80,1200]]), np.array([[[12,0,0], [0, 15,0], [0,0,100]], [[12,0,0], [0, 15,0], [0,0,100]], [[12,0,0], [0, 15,0], [0,0,100]]]), np.array([1.0/3, 1.0/3, 1.0/3])
realmeans, realcov, realpik = np.array([[22, 17], [85 , 150], [85, 15], [850, 150], [8500, 1500]]), \
                              np.array([[[1.2,0], [0, 1.5]], [[1.2,0], [0, 1.5]], [[1.2,0], [0, 1.5]], [[1.2,0], [0, 1.5]], [[1.2,0], [0, 1.5]]]), np.array([1.0/5, 1.0/5, 1.0/5, 1.0/5, 1./5])
#realmeans, realcov = 10 * realmeans,  10 * realcov
#realcov = 10*realcov
#X = samplingGMM(N,means_omega=realmeans, cov=realcov,pi=realpik)
X1 = multivariate_normal.rvs(mean=realmeans[0], cov=realcov[0], size=1500)
X2 = multivariate_normal.rvs(mean=realmeans[1], cov=realcov[1], size=1500)
X3 = multivariate_normal.rvs(mean=realmeans[2], cov=realcov[2], size=1500)
X4 = multivariate_normal.rvs(mean=realmeans[3], cov=realcov[3], size=1500)
X5 = multivariate_normal.rvs(mean=realmeans[4], cov=realcov[4], size=1500)

X = np.append(X1, X2, axis=0)
X = np.append(X, X3, axis=0)
X = np.append(X, X4, axis=0)
X = np.append(X, X5, axis=0)

logLikelihoodReal = logLikelihood(X, realmeans, realcov, realpik)


plt.scatter(X.T[0], X.T[1])
plt.show()

M_max = 32

max_silhouette_score = -4000
lis_silhouette_score = []
K = 1
y_better_labels = []
for k in range(2, M_max):
    cluster_K_means = KMeans(n_clusters=k)
    cluster_K_means.fit(X)
    y_labels = cluster_K_means.labels_

    actual_silhouette_score = silhouette_score(X, labels=y_labels, sample_size=int(len(X)*.8))
    if actual_silhouette_score > max_silhouette_score:
        max_silhouette_score = actual_silhouette_score
        K = k
        y_better_labels = y_labels

    lis_silhouette_score.append(actual_silhouette_score)


plt.plot(range(2, len(lis_silhouette_score) + 2), lis_silhouette_score)
plt.show()

piksampled = []
means = []
sigmas = []
N = len(X)

for k in range(K):
    Xk = X[np.where(y_better_labels == k)]
    Nk = len(Xk)
    pik = float(Nk)/N
    piksampled.append(pik)
    muk = np.mean(Xk, axis=0)
    sigmak = np.cov(Xk.T)
    means.append(muk)
    sigmas.append(sigmak)
    plt.scatter(Xk.T[0], Xk.T[1], label='Class ' + str(k + 1))
piksampled = np.array(piksampled)
means = np.array(means)
sigmas = np.array(sigmas)


print "Real pik"
print realpik
print "medias reales: "
print realmeans
print "covarianza reales"
print realcov


print "pik calculed"
print piksampled
print "medias sampled: "
print means
print "covarianza sampled"
print sigmas

plt.legend()
plt.show()






