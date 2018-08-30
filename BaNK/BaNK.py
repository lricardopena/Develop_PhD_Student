import numpy as np
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import multivariate_normal
from scipy.stats import invwishart, invgamma
from scipy.special import gamma
from scipy.special import gammaln
from scipy.stats import norm


def cholupdate(R, x, sign):
    p = np.size(x)
    x = x.T
    for k in range(p):
        if sign == '+':
            r = np.sqrt(R[k, k]**2 + x[k]**2)
        else:
            r = np.sqrt(R[k, k]**2 - x[k]**2)
        c = r/R[k, k]
        s = x[k]/R[k, k]
        R[k,k] = r
        if sign == '+':
          R[k, k+1 : p] = (R[k, k+1:p] + s*x[k + 1 : p])/c
        elif sign == '-':
          R[k,k+1:p] = (R[k,k+1:p] - s*x[k+1:p])/c
        x[k+1:p]= c*x[k+1:p] - s*R[k, k+1:p]
    return R


def scaledinvchisquare(niu, tausquare):
    return invgamma.rvs(a=niu/2.,scale=niu*tausquare/2., size=1)[0]


def returnBeta_sigma(X, Y, k):
    N = len(X)
    Q, R = np.linalg.qr(X)
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Rinv = np.linalg.inv(R + 0.0001 * np.identity(R.shape[0]))

    # if exist any nan or inf or -inf
    if np.any(np.isinf(Rinv)) or np.any(np.isnan(Rinv)):
        Rinv = np.nan_to_num(Rinv)
    # Vbeta = np.linalg.inv(X.T.dot(X))
    Vbeta = Rinv.dot(Rinv.T)
    # betahat = Vbeta.dot(X.T.dot(Y))
    betahat = Rinv.dot(Q.T.dot(Y))
    sSquare = 1. / (N - k) * (Y - X.dot(betahat)).T.dot(Y - X.dot(betahat))
    try:
        sigma = scaledinvchisquare(N - k, sSquare)
    except ValueError:
        print("stop")

    variance = Vbeta * (sigma ** 2)
    if np.any(np.isinf(variance)) or np.any(np.isnan(variance)):
        #variance = np.nan_to_num(variance)
        variance[variance == -np.inf] = -999999999
        variance[variance == np.inf] = 999999999
        variance[np.isnan(variance)] = 0.01
    #if np.any(np.isnan(variance)):
    #    index = []
    #    isnanVariance = np.isnan(variance)
    #    for i in range(variance.shape[0]):
    #        for j in range(variance.shape[0]):
    #            if isnanVariance[i][j]:
    #              index.append((i,j))
    try:
        betaProposal = np.random.multivariate_normal(betahat, variance)
        return betaProposal, sigma
    except:
        print("stop")
        #Checar cual de las dos
        #return betahat, sigma
        return Vbeta, sigma


def inversechisqu(niu, tausquare):
    while True:
        value = invgamma.rvs(a=niu / 2., scale=niu * tausquare / 2., size=1)[0]
        if not(np.isnan(value) or np.isinf(value)):
            return value


def compute_number_of_clusters(X, maximun_number_of_clusters, number_cross_validation):
    estimated_number_of_clusters = []
    mean, variance = 0, 0

    for i in range(number_cross_validation):
        clusters_and_score = [1, 0]

        silhouette_score_list = []
        for number_of_clusters in range(2, maximun_number_of_clusters):
            clusterer = KMeans(n_clusters=number_of_clusters)
            labels = clusterer.fit_predict(X)
            score_silhouette = silhouette_score(X, labels)
            if score_silhouette > clusters_and_score[1]:
                clusters_and_score[0] = number_of_clusters
                clusters_and_score[1] = score_silhouette
            silhouette_score_list.append(score_silhouette)
            mean = clusterer.cluster_centers_

        estimated_number_of_clusters.append(clusters_and_score[0])
    return int(np.round(np.sum(estimated_number_of_clusters)/float(number_cross_validation),0))


def samplingNew_muk_Sigmak(Psi0, kapa0, mu0, niu0):
    newsigma = invwishart.rvs(np.linalg.inv(Psi0), niu0, size=1)
    newMean = multivariate_normal.rvs(mu0, newsigma/kapa0)

    return newsigma, newMean


def samplingNew_muk_Sigmak_jeffreysPrior(Y):
    #invS = np.linalg.inv(np.cov(Y))
    invS = 1./np.cov(Y)
    mean = np.mean(Y)
    n = len(Y)
    #newsigma = invwishart.rvs(invS, n-1, size=1)
    niu = n-1
    tausquare = invS
    newsigma = invgamma.rvs(a=niu/2., scale=niu*tausquare/2., size=1)[0]
    newMean = multivariate_normal.rvs(mean, newsigma /n)

    if newsigma <= 0:
        print ("stop")

    return newsigma, newMean


#Zj indicates which component the random frenquency Wj is drawn from.
def samplingZj(mus, Z, sigmas, W,Y, M, alpha, ):
    Zk = np.array(Z)
    for j in range(M):
        p = []
        K = len(mus)
        for k in range(K):
            mk = len(np.where(Zk == k)[0])
            if k == Z[j]:
                mk -= 1

            if mk > 0:
                actualp = np.log(float(mk)) - np.log(M - 1 + alpha) + norm.logpdf(W[j], mus[k], sigmas[k])
                p.append(actualp)
            else:
                sigmak, muk = samplingNew_muk_Sigmak_jeffreysPrior(Y)
                p.append(float(alpha) / (M - 1 + alpha) * norm.logpdf(muk, sigmak, W[j]))
        #append the new means
        sigmak, muk = samplingNew_muk_Sigmak_jeffreysPrior(Y)
        try:
            p.append(float(alpha) / (M - 1 + alpha) * norm.logpdf(muk, sigmak, W[j]))
        except:
            p.append(float(alpha) / (M - 1 + alpha) * multivariate_normal.pdf(muk, sigmak, W[j]))

        paux = np.array(p) / np.sum(p)
        p = (1. / paux) / np.sum(1. / paux)
        actualp = 0
        u = np.random.uniform(size=1)[0]
        for k in range(K+1):
            actualp += p[k]
            if actualp >= u:
                if Z[j] != k:
                    Z[j] = k
                if k >= K:
                    mus = list(mus)
                    mus.append(muk)
                    mus = np.array(mus)
                    sigmas = list(sigmas)
                    sigmas.append(sigmak)

                    sigmas = np.array(sigmas)
                break


    if len(np.unique(Z)) < len(mus):
        Z, mus, sigmas = acomodateElementsInZ(Z, mus, sigmas, 0)

    return Z, mus, sigmas

def acomodateElementsInZ(Z, mus, sigmas, number_correct):
    newZ = list(Z)
    Znp = np.array(Z)
    for k in range(number_correct, len(mus)):
        Zk = np.where(Znp == k)[0]
        mk = len(Zk)
        if mk == 0:
            mus = np.delete(mus, k, 0)
            sigmas = np.delete(sigmas, k, 0)
            for j in range(M):
                if newZ[j] > k:
                    newZ[j] -= 1
            return acomodateElementsInZ(newZ, mus, sigmas, k)
    return newZ, mus, sigmas


def samplingMukAndSigmaK(Z, W, Psi0, kapa0, niu0, mu0, K):
    newSigmas = []
    newMeans = []
    Znp = np.array(Z)
    for k in range(K):
        Zk = np.where(Znp == k)[0]
        mk = len(Zk)
        if mk >= 1:
            if mk == 1:
                Sk = 0.0001
            else:
                Sk = np.cov(W[Zk])
            meanWk = np.mean(W[Zk])
            # Psik = Psi0 + Sk + float(kapa0*mk)/(kapa0 + mk)*(meanWk - mu0).dot((meanWk - mu0).T)
            Psik = Psi0 + Sk + float(kapa0 * mk) / (kapa0 + mk) * (meanWk - mu0) ** 2
            niuk = niu0 + mk
            # sigmak = invwishart.rvs(Psik, niuk,size=1)
            sigmak = inversechisqu(Psik, niuk)
            newSigmas.append(sigmak)
            kapak = kapa0 + mk
            mean = float(kapa0 * mu0 + mk * meanWk) / (kapa0 + mk)
            muk = multivariate_normal.rvs(mean, 1. / kapak * sigmak, size=1)
            newMeans.append(muk)
        else:
            for j in range(M):
                if Z[j] > k:
                    Z[j] -= 1
            Zk = np.where(Znp == k)[0]
            mk = len(Zk)
            if mk >= 1:
                if mk == 1:
                    Sk = 0.0001
                else:
                    Sk = np.cov(W[Zk])
                meanWk = np.mean(W[Zk])
                # Psik = Psi0 + Sk + float(kapa0*mk)/(kapa0 + mk)*(meanWk - mu0).dot((meanWk - mu0).T)
                Psik = Psi0 + Sk + float(kapa0 * mk) / (kapa0 + mk) * (meanWk - mu0) ** 2
                niuk = niu0 + mk
                # sigmak = invwishart.rvs(Psik, niuk,size=1)
                sigmak = inversechisqu(Psik, niuk)
                newSigmas.append(sigmak)
                kapak = kapa0 + mk
                mean = float(kapa0 * mu0 + mk * meanWk) / (kapa0 + mk)
                muk = multivariate_normal.rvs(mean, 1. / kapak * sigmak, size=1)
                newMeans.append(muk)

    return np.array(newMeans), np.array(newSigmas)


def Psi(X, W, Wj, j):
    PsiX = []
    Wtmp = W.copy()
    Wtmp[j] = Wj
    for x in X:
        #argument = Wtmp.dot(x).T[0]
        argument = Wtmp.dot(x)
        psix = np.concatenate((np.cos(argument), np.sin(argument)))
        PsiX.append(psix)
    return 1./np.sqrt(len(W)) * np.array(PsiX)


def generateRowWj(Wj, X, W, j):
    M = len(W)
    irow = np.zeros(2 * M)
    i_plus_M_row = np.zeros(2 * M)

    phi_new = 1./np.sqrt(M) * phiFunction(X, Wj).T
    for m in range(M):
        if m == j:
            irow[j] = phi_new[0].dot(phi_new[0])
            irow[j + M] = phi_new[0].dot(phi_new[1])
        else:
            phi_actual = 1./np.sqrt(M) * phiFunction(X, W[m]).T
            irow[m] = phi_new[0].dot(phi_actual[0])
            irow[m + M] = phi_new[0].dot(phi_actual[1])

        if m == j:
            i_plus_M_row[j] = phi_new[0].dot(phi_new[1])
            i_plus_M_row[j + M] = phi_new[1].dot(phi_new[1])
        else:
            phi_actual = 1./np.sqrt(M) * phiFunction(X, W[m]).T
            i_plus_M_row[m] = phi_new[1].dot(phi_actual[0])
            i_plus_M_row[m + M] = phi_new[1].dot(phi_actual[1])

    return irow, i_plus_M_row


def aceptanceRatio(Y, X, W, Wj, j, a0, b0, mu0, oldBeta, oldSigma, PsiX, oldL, oldLInverse, oldLLogDeterminant):
    W_new = W.copy()
    W_new[j] = Wj

    n = 3.

    PsiX_new = phiFunction(X, W_new)

    mu_beta, sigma = returnBeta_sigma(PsiX_new, Y, 1)
    if np.any(np.isnan(mu_beta)) or np.any(np.isnan(sigma)):
        return 0
    # lamda0 = 1. / 0.0001 * np.identity(2 * M)
    lamda0 = 1. / sigma * np.identity(2 * M)

    lambdan = PsiX_new.T.dot(PsiX_new) + lamda0
    L = np.linalg.cholesky(lambdan)



    j_row_new, j_plus_M_row_new = generateRowWj(Wj, X, W, j)
    j_row_old, j_plus_M_row_old = generateRowWj(W[j], X, W, j)

    #xx = np.zeros_like(lambdan)
    #xx[:, j] = j_row_old - j_row_new
    #xx[:, j + M] = j_plus_M_row_old - j_plus_M_row_new
    #xx[j, :] = j_row_old - j_row_new
    #xx[j + M, :] = j_plus_M_row_old - j_plus_M_row_new

    #xx = (xx - lamda0) + 1. / oldSigma * np.identity(2 * M)

    #L_try = cholupdate(L, xx, "+")
    try:
        Linverse = np.linalg.inv(L)
        # This is because A^-1 == (L^-1)^T(L^-1)
        mun = (Linverse.T.dot(Linverse)).dot(lamda0.dot(mu_beta) + PsiX_new.T.dot(Y))
    except:
        Linverse = np.linalg.inv(L + 0.0001 * np.identity(L.shape[0]))
        mun = (Linverse.T.dot(Linverse)).dot(lamda0.dot(mu_beta) + PsiX_new.T.dot(Y))

    mu0 = np.array([mu0]*len(mun))
    an = a0 + n/2.
    #an = a0 + len(X)/2. #with N degrees of freedom
    #using cholesky decomposition we have x.T * A * x = |x.T * L|**2, where L is cholesky decomposition such that A = L * L.T.conj()
    #bn = b0 + 1. / 2 * (Y.T.dot(Y) + mu0.T.dot(lamda0).dot(mu0) - mun.T.dot(lambdan).dot(mun))

    bn = b0 + 1. / 2 * (Y.T.dot(Y) + mu0.T.dot(lamda0).dot(mu0) - np.linalg.norm(mun.T.dot(L))**2) #using cholensky decomposition
    # bn = b0 + 1. / 2 * (Y.T.dot(Y) + mu0.T.dot(lamda0).dot(mu0) - mun.T.dot(lambdan).dot(mun))
    # this because, (mun.T.dot(lambdan).dot(mun) == np.linalg.norm(mun.T.dot(L))**2)


    logdeterminant0 = 2 * M * np.log(1./sigma)
    siglamdan, Llogdeterminant = np.linalg.slogdet(L)

    # its two times because is the cholesky decomposition
    logpProposal = (gammaln(an) - gammaln(a0)) + (a0*np.log(b0) - an * np.log(np.abs(bn))) + 1./2 * (logdeterminant0 - (Llogdeterminant*2) )
    #pProposal = gamma(an)/gamma(a0) * (np.power(b0, a0)/np.power(bn, an)) * np.sqrt(np.linalg.det(lamda0)/np.linalg.det(lambdan))

    #PsiX = Psi(X, W, W[j], j)

    lamda0 = 1. / oldSigma * np.identity(2 * M)

    mun = (oldLInverse.T.dot(oldLInverse)).dot(lamda0.dot(oldBeta) + PsiX.T.dot(Y))
    an = a0 + n / 2.
    # an = a0 + len(X)/2. #with N degrees of freedom
    # using cholesky decomposition we have x.T * A * x = |x.T * L|**2, where L is cholesky decomposition such that A = L * L.T.conj()
    # bn = b0 + 1. / 2 * (Y.T.dot(Y) + mu0.T.dot(lamda0).dot(mu0) - mun.T.dot(lambdan).dot(mun))

    bn = b0 + 1. / 2 * (Y.T.dot(Y) + mu0.T.dot(lamda0).dot(mu0) - np.linalg.norm(
        mun.T.dot(oldL)) ** 2)  # using cholensky decomposition
    # bn = b0 + 1. / 2 * (Y.T.dot(Y) + mu0.T.dot(lamda0).dot(mu0) - mun.T.dot(lambdan).dot(mun))
    # this because, (mun.T.dot(lambdan).dot(mun) == np.linalg.norm(mun.T.dot(L))**2)

    logdeterminant0 = 2*M * np.log(1./oldSigma)

    # its two times because is the cholesky decomposition
    logpPrevious = (gammaln(an) - gammaln(a0)) + (a0 * np.log(b0) - an * np.log(np.abs(bn))) + 1. / 2 * (
            logdeterminant0 - (oldLLogDeterminant * 2))

    result = logpProposal - logpPrevious
    if result < 0:
        p = np.exp(result)
        u = np.random.uniform(size=1)[0]
        if u <= p:
            # If accept, then return the new values
            return W_new, mu_beta, sigma, PsiX_new, L, Linverse, Llogdeterminant

        return W, oldBeta, oldSigma, PsiX, oldL, oldLInverse, oldLLogDeterminant

    # because e^x >=1 when x >=0, so we accept
    return W_new, mu_beta, sigma, PsiX_new, L, Linverse, Llogdeterminant


def samplingWj_regresion(Y, X, W, mus, sigmas, Z, M, a0, b0, mu0):
    PsiX = phiFunction(X, W)
    beta, sigma = returnBeta_sigma(PsiX, Y, 1)
    L = np.linalg.cholesky(PsiX.T.dot(PsiX) + 1./sigma * np.identity(2 * M))
    Linverse = Linverse = np.linalg.inv(L)
    signo, LlogDeterminant = np.linalg.slogdet(L)

    for j in range(M):
        wProposal = multivariate_normal.rvs(mus[Z[j]], cov=sigmas[Z[j]], size=1)
        W, beta, sigma, PsiX, L, Linverse, LlogDeterminant = \
            aceptanceRatio(Y, X, W, wProposal, j, a0, b0, mu0, oldBeta=beta, oldSigma=sigma, PsiX=PsiX, oldL=L,
                           oldLInverse=Linverse, oldLLogDeterminant=LlogDeterminant)
    return W


def samplingGMM(N, means, cov, pi):
    sampling = np.zeros((N, 1))
    for i in range(N):
        randomNumber = np.random.uniform()
        current = 0
        index = 0
        for actualpi in pi:
            current += actualpi
            if current > randomNumber:
                xi = norm.rvs(loc=means[index], scale=cov[index], size=1)
                sampling[i] = xi
                break
            index += 1

    return sampling


def GEM(alpha, lenght):
    pi = []
    betas = []
    for i in range(0, lenght-1):
        betak = beta.rvs(1,alpha)
        pik = 1
        for betal in betas:
            pik *= (1-betal)
        betas.append(betak)
        pik *= betak
        pi.append(pik)
    pi.append(1-np.sum(pi))

    return np.array(pi)


def phiFunction(X, w):
    if isinstance(w, float):
        argument = X.reshape((X.shape[0], 1)) * w
        resultCos = np.cos(argument)
        resultSin = np.sin(argument)
        resultPhi = np.column_stack((resultCos, resultSin))

        return 1./np.sqrt(resultPhi.shape[1]/2) * resultPhi

    argument = X.reshape((X.shape[0], 1)).dot(w.reshape(-1, w.shape[0]))
    resultCos = np.cos(argument)
    resultSin = np.sin(argument)
    resultPhi = np.column_stack((resultCos, resultSin))

    return 1./np.sqrt(resultPhi.shape[1]/2) * resultPhi


def phi_xi(x, w):
    argument = w.dot(x).T[0]

    return np.concatenate((np.cos(argument), np.sin(argument)))


def f(X):
    means = []
    for x in X:
        means.append(phi_xi(x, omegas))

    means = np.array(means).dot(beta)
    Y = np.random.multivariate_normal(mean=means, cov=np.identity(len(means)), size=1)[0]
    return Y


def trueKernel(X):
    return np.exp(-1.0/8 * X**2)*(1.0/2 + 1.0/2 * np.cos(3.0/4*np.pi*X))


def BaNKKernel(X, means, sigmas, pik):
    value = np.zeros_like(X)
    for i in range(len(means)):
        value += pik[i]*np.exp(-1./(2*sigmas[i]) * X**2)* np.cos(means[i]*X)
    return value


def generated_exits(X, W, beta, sigma):
    Y = []
    PsiX = phiFunction(X, W)
    for Psixi in PsiX:
        yi = np.random.normal(loc=Psixi.T.dot(beta), scale=sigma, size=1)[0]
        Y.append(yi)
    return np.array(Y)


def getPik(Z):
    Znk = np.array(Z)
    N = float(len(Z))
    pik = np.zeros((len(np.unique(Z))))
    for k in range(len(pik)):
        pik[k] =  len(np.where(Znk == k)[0])/N
    return pik


def functionf(Xi, means, cov, pik):
    Yi = []
    for x in Xi:
        y = 0
        for i in range(len(means)):
            y += pik*  np.random.normal(means[i], cov[i], size=1)[0]

        Yi.append(y)
    return np.array(Yi)

# means, cov, pik = np.array([0,2.*np.pi/6, 3.0/4* np.pi]), np.array([1.0/16, 1.0/16, 1.0/16]), np.array([1./3,1./3,1./3])
means, cov, realpik = np.array([0, 3. * np.pi / 4]), np.array([1.0 / 2**2, 1.0 / 2**2]), np.array([1. / 2, 1. / 2])
N = 1000
M = 250
sigma = 0.0001
Xi = norm.rvs(loc=0, scale=4, size=N)
omegas = samplingGMM(N=M, means=means, cov=cov, pi=realpik)
beta = np.array(multivariate_normal.rvs(mean=np.zeros(2 * M), cov = np.identity(2 * M), size=1))
# beta = np.random.normal(loc=0, scale=1, size= 2 * M)
# beta = np.ones(2*M)
Yi = f(Xi)
#hiperparameters
number_of_rounds = 5000
a0 = 0.01
b0 = 0.01
alpha = 1
mu0 = np.mean(Yi)
niu0 =  N-1
#Psi0 = np.linalg.inv(np.cov(Yi))
Psi0 = 1./np.cov(Yi)
kapa0 = 0.00001
#End of hyperparameters


#Xi = np.linspace(-10,10,1000)
#real = trueKernel(Xi)
#formed = BaNKKernel(Xi, means, 1./cov, realpik)
#plt.plot(Xi, real)
#plt.show()
#plt.plot(Xi, formed)
#plt.show()


d = 1
initial_K = 10
mus = np.random.random((d, initial_K))[0]
Z = list(np.random.randint(0, initial_K, size=M))
sigmas = np.random.random((d, initial_K))[0]

W = np.random.random((d, M))[0]
for i in range(number_of_rounds):
    Z, mus, sigmas = samplingZj(mus, Z, sigmas, W, Yi, M, alpha)
    mus, sigmas = samplingMukAndSigmaK(Z, W, Psi0, kapa0, niu0, mu0, len(mus))
    W = samplingWj_regresion(Yi, Xi, W, mus, sigmas, Z, M, a0, b0, mu0)


pik_get = getPik(Z)
Xi = np.linspace(-10, 10, 1000)
plt.plot(Xi, trueKernel(Xi), label='True kernel')
plt.plot(Xi, BaNKKernel(Xi, mus, sigmas, pik_get), label='BaNK')
plt.legend()
plt.show()

PsiX = phiFunction(Xi, W)
mu_beta, sigma = returnBeta_sigma(PsiX, Yi, 1)
Y_i_generated = generated_exits(Xi, W, beta, sigma)
print ("something")
#Need compare Yi_generated with Yi