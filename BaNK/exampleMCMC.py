import numpy as np
import scipy as sc
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import multinomial
from scipy.stats import wishart
import matplotlib.pyplot as plt
import math
import sys

Mmax = 32


def returnCovarianceMatrix(lambdaInverse):
    lambdaInverse = np.array(sorted(1./lambdaInverse, reverse=True))
    return A.dot(np.diag(lambdaInverse)).dot(A)

def InitializeZ(pi, means, lambdaInverse, X):
    N = len(X)
    M = len(means)
    Zgroups = []
    u = np.random.uniform(size=N)
    sigmaInverse = []
    sigmas = []
    constants = []
    for m in range(0,M):
        covariance = np.dot(np.dot(A, np.diag(1.0 / np.array(sorted(lambdaInverse[m])))), A)
        sigmas.append(covariance)
        sigmaInverse.append(np.linalg.inv(covariance))
        constants.append(np.log(pi[m]) + np.log(np.power(np.linalg.det(covariance), -1.0/2)))
        Zgroups.append([])

    sigmaInverse = np.array(sigmaInverse)
    constants = np.array(constants)
    for n in range(0,N):
        p = []
        for s in range(0,M):
            p.append( -1*(constants[s] +
                     -1.0/2 * np.dot(np.dot((X[n] - means[s]).T, sigmaInverse[s]),(X[n] - means[s]))))

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



def samplingGMM(N, means, cov, pi):
    sampling = np.zeros((N, means.shape[1]))
    #sampling = np.zeros((N, 1)) #1-Dimension
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

def priorM(M):
    if M <= Mmax:
        return 1.0/Mmax
    else:
        return 0


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

#Simultaed mixted weghts using stick break GEM, move type a
def updateMixingWeights(delta, Z):
    k = len(Z)
    q = np.zeros(k)
    u = np.zeros(k)
    nk = []
    for zk in Z:
        nk.append(len(zk))
    nk = np.array(nk)
    alphaVect = np.tile(delta, k) + nk
    for j in range(0, k - 1):
        u[j] = beta.rvs(alphaVect[j], np.sum(alphaVect[j+1:]))
        q[j] = u[j]
        if j > 0:
            for i in range(0,j):
                q[j] = q[j]* (1 - u[i])
    q[k-1] = 1 - np.sum(q)
    return q
    #maybe this is not use because is a gibbs sampling


#Simulted mu and sigma, move type b
def updateMeansAndCovariances(r, xi, tau, ellInverse, Z, X, means, lambdaInverse, A, pi):
    M = len(means)
    D = X.shape[1]

    newMeans = np.array(means)
    newlamdaInverse = np.array(lambdaInverse)

    for m in range(0, M):
        if len(Z[m]) > 0:
            # Compute the new mean
            xmean = np.mean(X[Z[m]], axis=0)
            xiBar = (tau[m] * xi[m] + len(Z[m]) * xmean) / (tau[m] + len(Z[m]))
            # Get the new mean
            cov = returnCovarianceMatrix(newlamdaInverse[m])
            newMeans[m] = multivariate_normal.rvs(mean=xiBar, cov=1.0 / (len(Z[m]) + tau[m]) * cov, size=1)

            # Compute the new covariance
            alpha = r[m] + len(Z[m]) + 1
            Sm = (X[Z[m]] - newMeans[m]).T.dot(X[Z[m]] - newMeans[m])

            beta = tau[m] * (newMeans[m] - xi[m]).reshape((-1, 1)) * (newMeans[m] - xi[m]).reshape((1, -1)) + Sm
            for d in range(0, D):
                betaprime = ellInverse[m][d] + np.dot(np.dot(A[d], beta), A[d])

                newlamdaInverse[m][d] = gamma.rvs(alpha / 2.0, betaprime / 2.0, size=1)
            # End of compute new covariance




    # Is a gibbs sampling, so always accept
    return newMeans, newlamdaInverse


# Derive the missing data Z, move type c
def updateAlocation(pi, means, lambdaInverse, X, Z):
    N = len(X)
    M = len(means)
    Z = list(Z)
    Zgroups = []
    u = np.random.uniform(size = N)
    sigmaInverse = []
    sigmas = []
    constants = []
    for m in range(0,M):
        #verificar esta multiplicacion
        inverseCovariance = np.dot(np.dot(A, np.diag(sorted(lambdaInverse[m]))), A)
        sigmaInverse.append(inverseCovariance)
        covariance = np.dot(np.dot(A, np.diag(1.0 / np.array(sorted(lambdaInverse[m])))), A)
        sigmas.append(covariance)
        constants.append(pi[m] * np.power(np.linalg.det(covariance), -1.0/2))
        Zgroups.append([])

    sigmaInverse = np.array(sigmaInverse)
    sigmas = np.array(sigmas)
    constants = np.array(constants)
    for n in range(0, N):
        p = []
        for s in range(0, M):
            p.append( -( np.log(constants[s]) + (-1.0/2 * np.dot(np.dot((X[n] - means[s]).T, sigmaInverse[s]),(X[n] - means[s])))) )

        if not np.sum(p) == 0:
            p = np.array(p)
            p = np.sum(p) / p
            p = p / np.sum(p)
        else:
            p = np.tile([1./ M], M)

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


#Update hyperparameters xi, tau and ell, move type d
def updateHyperparameters(r, zeta, xi, tau, niu, ellInverse, rhoSquare, means, lambdaInverse, A):
    D = A.shape[1]
    M = len(tau)
    inverseRhoSquare = 1.0 / rhoSquare

    for m in range(0,M):
        for d in range(0,D):
            ellInverse[m][d] = gamma.rvs((1 + r[m])/2.0, (lambdaInverse[m][d] + zeta[d])/2.0, size=1)

        inverseSigmam = np.linalg.inv(returnCovarianceMatrix(lambdaInverse[m]))
        sigmaHat = np.linalg.inv(inverseRhoSquare * np.identity(D) + tau[m] * inverseSigmam)
        newmean = np.dot(sigmaHat, inverseRhoSquare * niu + tau[m] * np.dot(inverseSigmam, means[m]))

        if np.isnan(sigmaHat).any():
            sigmaHat = 0.0001 * np.identity(sigmaHat.shape[0])
        try:
            xi[m] = multivariate_normal.rvs(mean=newmean, cov=sigmaHat, size=1)
        except ValueError:
            print "stop"
        alpha = D + 1
        beta = 1.0 / rhoSquare + np.dot(np.dot((means[m] - xi[m]), inverseSigmam), (means[m] - xi[m]))
        tau[m] = gamma.rvs(alpha/2.0, beta/2.0, size=1)

    #gibbs sampling, so we accept
    return xi, tau, ellInverse

#Split one component or combine two components into one, move type e
def splitOrCombineTwoComponents(pi, means, lambdaInverse, A, Z, delta, X, tau, niu, rhoSquare, ellInverse, xi, r):
    u = np.random.uniform(size=1)
    M = len(pi)

    bk = 0.5

    if M == 1:
        bk = 1
    elif M == Mmax:
        bk = 0

    if u < bk:
        return splitOneComponent(pi=pi, means=means, lambdaInverse=lambdaInverse, A=A, Z=Z, delta=delta, X=X, tau=tau, niu=niu, rhoSquare=rhoSquare, ellInverse=ellInverse, xi=xi, r=r)

    return combineTwoComponentsIntoOne(pi=pi, means=means, lambdaInverse=lambdaInverse, A=A, Z=Z, delta=delta, X=X, tau=tau, niu=niu, rhoSquare=rhoSquare, ellInverse=ellInverse, xi=xi, r=r)

def splitOneComponent(pi, means, lambdaInverse, A, Z, delta, X, tau, niu, rhoSquare, ellInverse, xi, r):

    Z = list(Z)
    M = len(pi)
    D = means.shape[1]
    u = np.random.uniform(size=1)
    p = 0
    k = 0
    bM = 0.5
    dM = 1 - bM
    palloc = 1.0 / M
    for k in range(0, M):
        p += 1.0 / M
        if p > u:
            break


    #If nothing to split
    if len(Z[k]) == 0:
        return pi, means, lambdaInverse, xi, tau, ellInverse, Z, r

    #Create the new hyperparameters
    tauj = gamma.rvs(1.0 / 2, 1.0 / rhoSquare * 1.0 / 2.0, size=1)[0]
    tauk = gamma.rvs(1.0 / 2, 1.0 / rhoSquare * 1.0 / 2.0, size=1)[0]
    xij = multivariate_normal.rvs(niu, rhoSquare * np.identity(D), size=1)
    xik = multivariate_normal.rvs(niu, rhoSquare * np.identity(D), size=1)
    ellInversej = gamma.rvs(1.0 / 2, 1.0 / rhoSquare * 1.0 / 2, size=D)
    ellInversek = gamma.rvs(1.0 / 2, 1.0 / rhoSquare * 1.0 / 2, size=D)
    rj = r[M-1]
    rk = r[M-1]
    alpha = float(beta.rvs(1, 1, size=1))
    newPij = alpha * pi[k]
    newPik = (1 - alpha) * pi[k]

    determinantOfJacobian = 1
    q = beta(1, 1).logpdf(alpha)

    sumofEigenvalues = 0
    newlambdaInversej = []
    newlambdaInversek = []
    for d in range(0,D):
        ud = float(beta.rvs(2, 2, size=1))
        q += beta(2,2).logpdf(ud)

        lambdakd = 1.0 / lambdaInverse[k][d]
        sumofEigenvalues += np.sqrt(lambdakd) * ud * A[d]

        betad = float(beta.rvs(1, 1, size=1))
        q += beta(1,1).logpdf(betad)

        newlambdaInversejd = betad * (1 - ud**2) * (pi[k] / newPij) * lambdakd
        newlambdaInversejd = 1. / newlambdaInversejd
        newlambdaInversej.append(newlambdaInversejd)
        newlambdaInversekd = (1 - betad) * (1 - ud**2) * (pi[k] / newPik) * lambdakd
        newlambdaInversekd = 1. / newlambdaInversekd
        newlambdaInversek.append(newlambdaInversekd)

        #Compute the jacobian
        determinantOfJacobian *= np.power(lambdakd,3.0/2) * (1 - ud**2)

    newlambdaInversej = np.array(newlambdaInversej)
    newlambdaInversek = np.array(newlambdaInversek)


    newMeanj = means[k] - np.sqrt(newPik / newPij) * sumofEigenvalues
    newMeank = means[k] + np.sqrt(newPij / newPik) * sumofEigenvalues

    #split the number of elements in Zk
    newElementsInZj = []
    newElementsInZk = []
    sigmak = returnCovarianceMatrix(newlambdaInversek)
    sigmaj = returnCovarianceMatrix(newlambdaInversej)

    pnk = np.log(newPik) + sc.stats.multivariate_normal.logpdf(x=X[Z[k]], mean=newMeank, cov=sigmak)
    pnj = np.log(newPij) + sc.stats.multivariate_normal.logpdf(x=X[Z[k]], mean=newMeanj, cov=sigmaj)
    #pnk = np.log(newPik) - 1./2 * np.log(np.linalg.det(sigmak)) - 1.0/2 * (X[Z[k]] - newMeank).dot(np.linalg.inv(sigmak)).dot((X[Z[k]] - newMeank).T).diagonal()
    #pnj = np.log(newPij) - 1./2 * np.log(np.linalg.det(sigmaj)) - 1.0/2 * (X[Z[k]] - newMeanj).dot(np.linalg.inv(sigmaj)).dot((X[Z[k]] - newMeanj).T).diagonal()


    for i, pnkj in enumerate(np.column_stack((pnk,pnj))):
        p = 1 - pnkj[0] / np.sum(pnkj)
        u = np.random.uniform(size = 1)[0]
        if u < p:
            newElementsInZk.append(Z[k][i])
        else:
            newElementsInZj.append(Z[k][i])

    # Now we compute the  probability of Acept a combine move
    sigmaoldk = np.dot(np.dot(A, np.diag(1.0 / np.array(sorted(lambdaInverse[k])))), A)

    firstExponent = float(np.dot(np.dot((newMeanj - xij).reshape(1, -1), tauj * np.linalg.inv(sigmaj)), newMeanj - xij) +
                np.dot(np.dot((newMeank - xik).reshape(1, -1), tauk * np.linalg.inv(sigmak)), newMeank - xik) -
                np.dot(np.dot((means[k] - xi[k]).reshape(1, -1), tau[k] * np.linalg.inv(sigmaoldk)), means[k] - xi[k]))


    secondExponent = 0
    resultOfMultiplier = 1
    for d in range(0, D):
        secondExponent += newlambdaInversej[d] * ellInversej[d] + newlambdaInversek[d] * ellInversek[d] - \
                          lambdaInverse[k][d] * ellInverse[k][d]



        lambdakd = 1.0/lambdaInverse[k][d]
        ellkd = 1.0/ellInverse[k][d]

        newlambdajd = 1.0/newlambdaInversej[d]
        newelljd = 1.0/ ellInversej[d]

        newlambdakd = 1.0/ newlambdaInversek[d]
        newellkd = 1.0/ ellInversek[d]

        numerator = lambdakd * np.power((newlambdajd * newelljd) /2.0,-rj/2.0)* np.power((newlambdakd * newellkd) /2.0,-rk/2.0)
        denominator = newlambdajd * newlambdakd * np.power((lambdakd * ellkd)/2.0,-r[k]/2.0)

        resultOfMultiplier *= (numerator / denominator)

    # delete the old values of k
    newZ = list(Z)
    # Delete the old value
    newPi = np.delete(pi, k, axis=0)
    newMeans = np.delete(means, k, axis=0)
    newLambdaInverse = np.delete(lambdaInverse, k, axis=0)
    newxi = np.delete(xi, k, axis=0)
    newtau = np.delete(tau, k, axis=0)
    newr = np.delete(r, k, axis=0)
    newellInverse = np.delete(ellInverse, k, axis=0)
    del newZ[k]

    # Append the new values j
    newPi = np.append(newPi, [newPij], axis=0)
    means = np.append(newMeans, [newMeanj], axis=0)
    lambdaInverse = np.append(newLambdaInverse, [np.array(newlambdaInversej)], axis=0)
    newZ.append(newElementsInZj)
    # Append the hyperparameter of j
    newtau = np.append(newtau, [tauj], axis=0)
    newxi = np.append(newxi, [xij], axis=0)
    newellInverse = np.append(newellInverse, [ellInversej], axis=0)
    newr = np.append(newr, [rj], axis=0)

    # Append the new values k
    newPi = np.append(newPi, [newPik], axis=0)
    newMeans = np.append(means, [newMeank], axis=0)
    newLambdaInverse = np.append(lambdaInverse, [np.array(newlambdaInversek)], axis=0)
    newZ.append(newElementsInZk)
    # Append the hyperparameter of k
    newtau = np.append(newtau, [tauk], axis=0)
    newxi = np.append(newxi, [xik], axis=0)
    newellInverse = np.append(newellInverse, [ellInversek], axis=0)
    newr = np.append(newr, [rk], axis=0)


    oldCovariances = []
    for m in range(0, M):
        oldCovariances.append(A.dot(np.diag(1.0 / np.array(sorted(lambdaInverse[m])))).dot(A))
    oldCovariances = np.array(oldCovariances)

    newCovariances = []
    for m in range(0, len(newPi)):
        newCovariances.append(A.dot(np.diag(1.0 / np.array(sorted(newLambdaInverse[m])))).dot(A))
    newCovariances = np.array(newCovariances)


    oldLogLikelihood = logLikelihood(Y=X, means=means, sigmas=oldCovariances, pik=pi, Z=Z)
    newLogLikelihood = logLikelihood(Y=X, means=newMeans, sigmas=newCovariances, pik=newPi, Z=newZ)
    valueLogLikelihood = newLogLikelihood - oldLogLikelihood

    R = priorM(M + 1) / priorM(M) * (M + 1)
    R *= (np.power(newPij,delta - 1 + len(newElementsInZj)) * np.power(newPik,delta - 1 + len(newElementsInZk)))
    R /= (np.power(pi[k], delta - 1 + len(newElementsInZj) + len(newElementsInZk)) * sc.special.beta(delta, M * delta))
    R *= (np.power(2*np.pi, - D/2.0) * np.power(np.linalg.det(1.0/tau[k] * sigmaoldk), 1.0/2) / (np.power(np.linalg.det(1.0/tauj * 1.0/tauk * np.dot(sigmaj, sigmak)),1.0/2)))
    R *= np.power(sc.special.gamma(r[k]/2.0)/(sc.special.gamma(rj/2.0) * sc.special.gamma(rk/2.0)), D)
    R *= resultOfMultiplier
    R *= np.exp(valueLogLikelihood - 1.0 / 2 * firstExponent - 1.0 / 2 * secondExponent)


    determinantOfJacobian = (np.power(pi[k],3*D + 1) /np.power(newPij * newPik, 3*D /2.0)) * determinantOfJacobian

    R = (R * dM)/(bM * palloc * np.exp(q)) * determinantOfJacobian


    probabilityOfAcept = np.array([1, R]).min()

    u = np.random.uniform(size=1)[0]


    if u < probabilityOfAcept:
        return newPi, newMeans, newLambdaInverse, newxi, newtau, newellInverse, newZ, newr

    return pi, means, lambdaInverse, xi, tau, ellInverse, Z, r

def combineTwoComponentsIntoOne(pi, means, lambdaInverse, A, Z, delta, X, tau, niu, rhoSquare, ellInverse, xi, r):
    M = len(pi)
    D = means.shape[1]
    u = np.random.uniform(size=1)
    Z = list(Z)
    bM = 0.5
    dM = 1 - bM
    palloc = 1.0 / (M - 1)
    p = 0
    i = 0
    for i in range(0, M - 1):
        p += 1.0/(M - 1)
        if p > u:
            break

    #We select the next component to mix
    j = i + 1

    newPiPrime = pi[i] + pi[j]
    newMean = (pi[i] * means[i] + pi[j] * means[j])/newPiPrime
    newLamdaInverse = []

    for d in range(0, D):
        newLamdaInversed = pi[i]/newPiPrime * lambdaInverse[i][d] + pi[j]/newPiPrime * lambdaInverse[j][d] + \
                           (pi[i] * pi[j])/(newPiPrime**2) * (means[i][d] - means[j][d])**2
        newLamdaInverse.append(newLamdaInversed)

    newPi = np.array(pi)
    newPi[i] = newPiPrime

    newMeans = np.array(means)
    newMeans[i] = newMean

    newLambdaInverse = np.array(lambdaInverse)
    newLambdaInverse[i] = np.array(newLamdaInverse)
    newZ = list(Z)
    if len(Z[i]) > 0 and len(Z[j]) > 0:
        newZ[i] = sorted(list(Z[i]) + list(Z[j]))
    elif len(Z[i]) > 0:
        newZ[i] = sorted(Z[i])
    else:
        newZ[i] = sorted(Z[j])


    newPi = np.delete(newPi, j, axis=0)
    newMeans = np.delete(newMeans, j, axis=0)
    newLambdaInverse = np.delete(newLambdaInverse, j, axis=0)
    del newZ[j]

    #update the hyperparameters
    newtau = np.delete(tau, j, axis=0)
    newxi = np.delete(xi, j, axis=0)
    newellInverse = np.delete(ellInverse, j, axis=0)
    newr = np.delete(r, j, axis=0)

    # Now we compute the  probability of Acept a combine move
    oldCovariances = []
    for m in range(0, M):
        oldCovariances.append(A.dot(np.diag(1.0 / np.array(sorted(lambdaInverse[m])))).dot(A))
    oldCovariances = np.array(oldCovariances)

    newCovariances = []
    for m in range(0, len(newPi)):
        newCovariances.append(A.dot(np.diag(1.0 / np.array(sorted(newLambdaInverse[m])))).dot(A))
    newCovariances = np.array(newCovariances)

    secondExponent = 0

    resultOfMultiplier = 1
    for d in range(0, D):
        secondExponent += lambdaInverse[i][d] * ellInverse[i][d] + \
                          lambdaInverse[j][d] * ellInverse[j][d] - newLambdaInverse[i][d] * newellInverse[i][d]


        lambdaid = 1.0 / lambdaInverse[i][d]
        ellid = 1.0 / ellInverse[i][d]

        lambdajd = 1.0 / lambdaInverse[j][d]
        elljd = 1.0 / ellInverse[j][d]

        newlambdaid = 1.0 / newLambdaInverse[i][d]
        newellid = 1.0 / newellInverse[i][d]

        numerator = newlambdaid * np.power(lambdajd * elljd, -r[j]/2.0) * np.power(lambdaid * ellid, -r[i]/2.0)
        denominator = lambdajd * lambdaid * np.power((newlambdaid * newellid)/2.0, - newr[i]/2.0)

        resultOfMultiplier *= (numerator / denominator)

    firstExponent = float( np.dot(np.dot((means[i] - xi[i]).reshape(1, -1), tau[i] * np.linalg.inv(oldCovariances[i])), means[i] - xi[i]) +
        np.dot(np.dot((means[j] - xi[j]).reshape(1, -1), tau[j] * np.linalg.inv(oldCovariances[j])), means[j] - xi[j]) -
        np.dot(np.dot((newMeans[i] - newxi[i]).reshape(1, -1), newtau[i] * np.linalg.inv(newCovariances[i])), newMeans[i] - newxi[i]) )

    oldLogLikelihood = logLikelihood(Y=X, means=means, sigmas=oldCovariances, pik=pi, Z=Z)
    newLogLikelihood = logLikelihood(Y=X, means=newMeans, sigmas=newCovariances, pik=newPi, Z=newZ)
    valueLogLikelihood = newLogLikelihood - oldLogLikelihood

    R = priorM(M+1) / priorM(M) * (M + 1)
    R *= (np.power(pi[i], delta - 1 + len(Z[i])) * np.power(pi[j], delta - 1 + len(Z[j])))
    R /= (np.power(newPi[i], delta - 1 + len(newZ[i]) ) * sc.special.beta(delta, M * delta))
    R *= (np.power(2 * np.pi, - D / 2.0) * np.power(np.linalg.det(1.0 / newtau[i] * newCovariances[i]), 1.0 / 2) / (
    np.power(np.linalg.det(1.0 / tau[i] * 1.0 / tau[j] * np.dot( oldCovariances[i], oldCovariances[j])), 1.0 / 2)))
    R /= np.power(sc.special.gamma(newr[i] / 2.0) / (sc.special.gamma(r[i] / 2.0) * sc.special.gamma(r[j] / 2.0)), D)
    R *= resultOfMultiplier


    determinantOfJacobian = computeJacobianOfCombineTwoComponents(pi[i], pi[j], means[i], means[j],
                                                                  1.0 / lambdaInverse[i], 1.0 / lambdaInverse[j])

    R = R * ( dM  ) / ( bM * palloc )

    R = 1.0/R

    R *= np.exp(-valueLogLikelihood + 1.0 / 2 * firstExponent + 1.0 / 2 * secondExponent) * determinantOfJacobian

    probabilityOfAcept = np.array([1, R]).min()

    u = np.random.uniform(size=1)[0]
    if u < probabilityOfAcept:
        return newPi, newMeans, newLambdaInverse, newxi, newtau, newellInverse, newZ, newr

    return pi, means, lambdaInverse, xi, tau, ellInverse, Z, r

def computeJacobianOfCombineTwoComponents(pii, pij, miui, miuj, gi, gj):
    new_pi = pii + pij
    D = miui.shape[0]
    jacobian = np.zeros((2*D + 1, 4*D + 2))
    jacobian[0][0], jacobian[0][1]  = 1,1 #derivate of piPrime with respect to pii and pij

    # derivates of new_gi
    #derivate of giprime with respect of pii
    derivateOfNewGiAgainstPii = 1.0/(new_pi**2) *(new_pi * gi + pij * np.array([res**2 for res in miui - miuj]))

    jacobian[1:D + 1, 0] = np.array(derivateOfNewGiAgainstPii)

    #derivate of giprime with respect of pij
    derivateOfNewGiAgainstPij = 1.0/(new_pi**2) *(new_pi * gj + pii * np.array([res**2 for res in miui - miuj]))
    jacobian[1:D + 1, 1] = np.array(derivateOfNewGiAgainstPij)

    #derivate of giprime with respect of gi
    derivateOfNewGiAgainstGi = pii/new_pi * np.ones((D, D))
        #We add the result to the jacobian
    columnsNumber = 2
    jacobian[1:D + 1, columnsNumber:columnsNumber + D] = np.array(derivateOfNewGiAgainstGi)

    # derivate of giprime with respect of gj
    derivateOfNewGiAgainstGj = pij / new_pi * np.ones((D, D))
        #We add the result to the jacobian
    columnsNumber = 2 +  D
    jacobian[1:D + 1, columnsNumber: columnsNumber + D] = np.array(derivateOfNewGiAgainstGj)

    #derivate of giprime with respect of mui
    derivateOfNewGiAganistmui = 2*pii*pij/(new_pi**2) * (np.ones((D,1)) * (np.transpose(miui - miuj)))
        #We add the result to the jacobian
    columnsNumber = 2 + 2 * D
    jacobian[1:D + 1, columnsNumber:columnsNumber + D] = np.array(derivateOfNewGiAganistmui)

    # derivate of giprime with respect of muj
    derivateOfNewGiAganistmuj = -2 * pii * pij / (new_pi**2) * (np.ones((D, 1)) * (np.transpose(miui - miuj)))
    columnsNumber = 2 + 3 * D
    jacobian[1:D + 1, columnsNumber:columnsNumber + D] = np.array(derivateOfNewGiAganistmuj)

    #derivatives of new mu

    derivateOfNewMuAgainstPii = 1.0/ new_pi * miui
    columnsNumber = 0
    jacobian[D + 1:2 * D + 1, columnsNumber] = np.array(derivateOfNewMuAgainstPii)


    derivateOfNewMuAgainstPij = 1.0 / new_pi * miuj
    columnsNumber = 1
    jacobian[D + 1:2 * D + 1, columnsNumber] = np.array(derivateOfNewMuAgainstPij)

    derivateOfNewMuAgainstMiui = pii / new_pi * np.identity(D)
    columnsNumber = 2 * D + 2
    jacobian[D + 1:2 * D + 1, columnsNumber:columnsNumber + D ] = np.array(derivateOfNewMuAgainstMiui)

    derivateOfNewMuAgainstMiui = pij / new_pi * np.identity(D)
    columnsNumber = 3 * D + 2
    jacobian[D + 1:2 * D + 1, columnsNumber:columnsNumber + D] = np.array(derivateOfNewMuAgainstMiui)


    determinantOfJacobian = np.linalg.det(np.dot(np.transpose(jacobian), jacobian))
    #determinantOfJacobian = np.linalg.det(np.dot(jacobian, np.transpose(jacobian)))
    if determinantOfJacobian <= 0:
        determinantOfJacobian = np.linalg.det(np.dot(jacobian, np.transpose(jacobian)))

    determinantOfJacobian = np.sqrt(determinantOfJacobian)
    if determinantOfJacobian <= 0:
        print "stop"
    return determinantOfJacobian

#Birth or death of an empty component, move type f
def birthOrDeathOfEmptyComponent(pi, means, lambdaInverse, Z, xi, tau, r, ellInverse, A, delta):
    bn = 0.5
    u = np.random.uniform(size=1)
    M = len(pi)
    dMplus1 = float(1 - bn)
    typeMove = ""
    if M == 1:
        #If only one multivariate, then the probability of death is 0
        bn = 1
        dMplus1 = 0.5
    elif M == Mmax:
        #If we get the maximun number of multivariate, then the probability of a new birth is 0
        bn = 0

    # compute the number of empty components
    numberOfEmptyComponents = 0
    for zn in Z:
        if len(zn) == 0:
            numberOfEmptyComponents += 1


    if u < bn: #birth of an empty component
        typeMove = "Birth empty component"
        piProposal, meansProposal, lambdaInverseProposal, xiProposal, tauProposal, ellInverseProposal, rProposal, ZProposal = \
            birthOfAnEmptyComponent(pi=pi, means=means, lambdaInverse=lambdaInverse, Z=Z, xi=xi, tau=tau, r=r, niu=niu,
                                    rhoSquare=rhoSquare, ellInverse=ellInverse, A=A)

        # Now we compute the aceptance probability R
        R = priorM(M + 1) / priorM(M)
        R *= 1.0 / sc.special.beta(delta, M * delta)
        R *= np.power(piProposal[M], delta - 1)
        R *= np.power(1 - piProposal[M], N + M * delta - M) * (M + 1)
        R *= dMplus1 / ((numberOfEmptyComponents + 1) * bn)
        R *= 1.0 / beta(1, M).pdf(piProposal[M])
        R *= np.power(1 - piProposal[M], M - 1)
    else: #death of an empty component
        typeMove = "death empty component"
        piProposal, meansProposal, lambdaInverseProposal, xiProposal, tauProposal, ellInverseProposal, rProposal, ZProposal, index = deathOfAComponent(
            pi=pi, means=means, lambdaInverse=lambdaInverse, Z=Z, xi=xi, tau=tau, r=r, ellInverse=ellInverse, delta=delta)

        # Now we compute the aceptance probability R^-1

        if ((numberOfEmptyComponents + 1) * bn) == 0:
            print "stop"

        R = priorM(M + 1) / priorM(M)
        R *= 1.0 / sc.special.beta(delta, M * delta)
        R *= np.power(pi[index], delta - 1)
        R *= np.power(1 - pi[index], N + M * delta - M) * (M + 1)
        try:
            R *= dMplus1 / ((numberOfEmptyComponents + 1) * bn)

            R *= 1.0 / beta(1, M).pdf(pi[index])
            R *= np.power(1 - pi[index], M - 1)

            R = np.power(R, -1)
        except ArithmeticError:
            R = 0 #Si hay una division por zero, entonces es 0, ya que estamos buscando la inversa

    u = np.random.uniform(size=1)[0]

    probabilityOfAceptance = np.array([1, R]).min()
    if u < probabilityOfAceptance:
        return piProposal, meansProposal, lambdaInverseProposal, xiProposal, tauProposal, ellInverseProposal, rProposal, ZProposal
    return pi, means, lambdaInverse, xi, tau, ellInverse, r, Z

def birthOfAnEmptyComponent(pi, means, lambdaInverse, Z, xi, tau, r, niu, rhoSquare, ellInverse, A):
    Z = list(Z)
    Z.append([])
    M = len(pi)
    D = means.shape[1]
    r = np.append(r, [r[M-1]], axis = 0)
    newpi = float(beta.rvs(1, M, size=1))
    pi = pi * (1 - newpi)

    #append the new hyperparameters
    xi = np.append(xi, [multivariate_normal.rvs(niu, rhoSquare * np.identity(D), size=1)], axis=0)
    tau = np.append(tau, gamma.rvs(1.0/2, 1.0/rhoSquare * 1.0/2.0, size=1), axis=0)
    ellInverse = np.append(ellInverse, [gamma.rvs(1.0 / 2, 1.0 / rhoSquare * 1.0 / 2, size=D)], axis=0)

    pi = np.append(pi, [newpi], axis=0)
    #Now we have to sampling the new means an covariance matrix

    #Append the new parameters
    newLambdaInverse = np.zeros(D)
    for d in range(0, D):
        newLambdaInverse[d] = gamma.rvs(r[M]/2, ellInverse[M][d]/2, size=1)[0]

    newLambdaInverse = np.array(newLambdaInverse)
    cov = np.dot(np.dot(A, np.diag(1.0 / np.array(sorted(newLambdaInverse)))), A)
    means = np.append(means, [multivariate_normal.rvs(mean=xi[M], cov=1.0 / tau[M] * cov, size=1)], axis=0)
    lambdaInverse = np.append(lambdaInverse, [newLambdaInverse], axis=0)

    return pi, means, lambdaInverse, xi, tau, ellInverse, r, Z

#We delete an empty component
def deathOfAComponent(pi, means, lambdaInverse, Z, xi, tau, r, ellInverse, delta):
    Z = list(Z)
    indexEmpty = []
    index = -1
    for j in range(0, len(Z)):
        if len(Z[j]) == 0:
            indexEmpty.append(j)
    numberOfEmptyComponents = len(indexEmpty)
    if numberOfEmptyComponents > 0:
        u = float(np.random.uniform(size=1)[0])
        p = 0.0
        for index in indexEmpty:
            p += 1.0/numberOfEmptyComponents
            if p > u:
                #Delete the parameters
                pi = np.delete(pi, index)
                pi = pi / np.sum(pi)
                means = np.delete(means, index, axis=0)
                lambdaInverse = np.delete(lambdaInverse, index, axis=0)
                del Z[index]

                #delete the hyperparameters
                r = np.delete(r, index, axis=0)
                xi = np.delete(xi, index, axis=0)
                tau = np.delete(tau, index, axis=0)
                ellInverse = np.delete(ellInverse, index, axis=0)
                break
    return pi, means, lambdaInverse, xi, tau, ellInverse, r, Z, index


#Return the log likelihood
def logLikelihood(Y, means, sigmas, pik, Z):
    resultLoglikelihood = 0
    pdfs = []


    for k in range(0, len(means)):
        pdfs.append(multivariate_normal(means[k], sigmas[k], allow_singular=True))


    k = 0
    for znk in Z:
        try:
            resultLoglikelihood += np.sum(np.log(pik[k]) + pdfs[k].logpdf(Y[znk]))
        except IndexError:
            print "stop"
        k += 1

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

def computeMatrices(lamdaInverse,A):
    cov = []
    M = lamdaInverse.shape[0]
    for m in range(0,M):
        cov.append((A.dot((np.diag(1.0/lambdaInverse[0]))).dot(A)))

    cov = np.array(cov)

    return cov

N = 1500
realmeans, realcov, realpik = np.array([[17.5, 5.5], [12.5,11], [3,8]]), np.array([[[1.2,0], [0, 1.5]], [[1.2,0], [0, 1.5]], [[1.2,0], [0, 1.5]]],), np.array([1.0/3, 1.0/3, 1.0/3])
#realmeans, realcov, realpik = np.array([[2200, 1700], [85,115], [1,3]]), np.array([[[12,0], [0, 15]], [[12,0], [0, 15]], [[12,0], [0, 15]]],), np.array([1.0/3, 1.0/3, 1.0/3])
#X = samplingGMM(N,means=realmeans, cov=realcov,pi=realpik)
X1 = multivariate_normal.rvs(mean=realmeans[0], cov=realcov[0], size=150)
X2 = multivariate_normal.rvs(mean=realmeans[1], cov=realcov[1], size=150)
X3 = multivariate_normal.rvs(mean=realmeans[2], cov=realcov[2], size=150)

X = np.append(X1, X2, axis=0)
X = np.append(X, X3, axis=0)
realZ = []
realZ.append(list(np.tile(0, 150)))
realZ.append(list(np.tile(1, 150)))
realZ.append(list(np.tile(2, 150)))

logLikelihoodReal = logLikelihood(X, realmeans, realcov,realpik, realZ)

#Set the hyperparameters

#M = np.random.randint(1, Mmax)
M = 8
r, delta, niu  = np.tile([4], M), 1, np.mean(X, axis=0)
rhoSquare = 0
for x in X:
    rhoSquare += np.linalg.norm(niu - x)**2
rhoSquare = 1.0/N * rhoSquare


xmean = np.mean(X, axis=0)
A = np.zeros((X.shape[1], X.shape[1]))

zeta = np.zeros(X.shape[1])
for x in X:
    zeta += [i**2 for i in x-niu]
    A += (x - xmean).reshape(-1,1) * (x - xmean)
zeta = 1.0/N * zeta
A = 1.0/N * A


#End set the hyperparameters

#proof of functions:
D = niu.shape[0]
pi = GEM(delta, M)
Z = multinomial.rvs(N, pi)



xi = multivariate_normal.rvs(mean=niu, cov=rhoSquare * np.identity(D), size = M)
tau = gamma.rvs(1.0/2, 1.0/rhoSquare * 1.0/2, size = M)
ellInverse = np.zeros((M, D))
for m in range(0, M):
    for d in range(0, D):
        ellInverse[m][d] = gamma.rvs(1.0/2, zeta[d], size = 1)[0]

means = []
lambdaInverse = []

for m in range(0, M):
    inverseLambdam = []
    for d in range(0, D):
        inverseLambdam.append(gamma.rvs(r[m] / 2.0, ellInverse[m][d] / 2.0, size=1)[0])
    inverseLambdam = np.array(inverseLambdam)
    cov = np.dot(np.dot(A, np.diag(1.0 / np.array(sorted(inverseLambdam)))), A)
    means.append(multivariate_normal.rvs(mean=xi[m], cov=1.0/tau[m] * cov, size=1))
    lambdaInverse.append(inverseLambdam)


means = np.array(means)
lambdaInverse = np.array(lambdaInverse)




Z = InitializeZ(pi=pi, means=means, lambdaInverse=lambdaInverse, X=X)
Z = updateAlocation(pi=pi, means=means, lambdaInverse=lambdaInverse, X=X, Z=Z)
#Begin
k = 1
for znk in Z:
    plt.scatter(X[znk].T[0],X[znk].T[1], label='Class ' + str(k))
    k += 1


plt.legend()
plt.show()


numberOfIterations = 200000

for i in range(0, numberOfIterations):
    pi, means, lambdaInverse, xi, tau, ellInverse, r, Z = adjustIndex(pi, means, lambdaInverse, xi, tau, ellInverse, r, Z)

    if i >= numberOfIterations - 90 :
        print computeMatrices(lambdaInverse, A)

    pi = updateMixingWeights(delta=delta, Z=Z)

    means, lambdaInverse = updateMeansAndCovariances(r=r, xi=xi, tau=tau, ellInverse=ellInverse, Z=Z, X=X, means=means,
                                                         lambdaInverse=lambdaInverse, A=A, pi=pi)

    Z = updateAlocation(pi=pi, means=means, lambdaInverse=lambdaInverse, X=X, Z=Z)


    xi, tau, ellInverse = updateHyperparameters(r=r, zeta=zeta, xi=xi, tau=tau, niu=niu, ellInverse=ellInverse, rhoSquare=rhoSquare,
                                                    means=means, lambdaInverse=lambdaInverse, A=A)

    pi, means, lambdaInverse, xi, tau, ellInverse, Z, r = splitOrCombineTwoComponents(pi=pi, means=means, lambdaInverse=lambdaInverse,
                                                                                          A=A, Z=Z, delta=delta, X=X, tau=tau, niu=niu,
                                                                                          rhoSquare=rhoSquare, ellInverse=ellInverse,
                                                                                          xi=xi, r=r)

    pi, means, lambdaInverse, xi, tau, ellInverse, r, Z = birthOrDeathOfEmptyComponent(pi=pi, means=means, lambdaInverse=lambdaInverse,
                                                                                           Z=Z, xi=xi, tau=tau, r=r, ellInverse=ellInverse,
                                                                                           A=A, delta=delta)

k=1
for znk in Z:
    plt.scatter(X[znk].T[0], X[znk].T[1], label='Class ' + str(k))
    k += 1


plt.legend()
plt.show()

print "stop"
