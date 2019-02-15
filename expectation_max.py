import numpy as np
from scipy.stats import multivariate_normal
import scipy.stats as ss

def em(dataset, clusters, tol=0.01, max_iter=10000000):
    if dataset is None:
        print('No data')
        return
    if clusters is None:
        print('No clusters')
        return

    # weight for each point to each cluster
    pis = [1/float(clusters) for _ in range(clusters)]
    pis = np.array(pis)

    # parameter means
    mus = [1 for _ in range(clusters)]
    mus = np.array(mus)
    mus = np.random.random(clusters)

    # covariance
    sigmas = [1 for _ in range(clusters)]
    sigmas = np.array(sigmas)
    sigmas = np.random.random(clusters)

    print(mus)
    print(sigmas)
    # likelihood variables
    ll_old = 0.0
    ll_new = 1.0


    iterations = 0
    while(iterations < max_iter):

        # E Step
        # r is probablity that a point belongs to a cluster
        r_ic = np.zeros((len(dataset), clusters))

        for i in range(len(dataset)):
            sum = 0
            for j in range(clusters):
                r_ic[i, j] = pis[j] * multivariate_normal(mus[j], sigmas[j]).pdf(dataset[i])
                sum += r_ic[i, j]
            r_ic[i] /= sum
            
        # M Step
        # compute m_c
        m_c = np.zeros(clusters)
        for i in range(len(m_c)):
            m_c[i] = np.sum(r_ic[:,i])

        # compute news pis
        for i in range(clusters):
            pis[i] = m_c[i] / np.sum(m_c)

        #compute new mus
        for i in range(clusters):
            mus[i] = 1/m_c[i] * np.sum(r_ic[:,i]) * np.sum(dataset)

        #computes new sigmas
        for i in range(clusters):
            temp = np.zeros(clusters)
            for j in range(len(dataset)):
                xMinusMu = dataset[j] - mus
                #print("x-mu: {}".format(xMinusMu))
                #print("x-mu dot: {}".format(np.dot((xMinusMu).T, (xMinusMu))))
                temp[i] += r_ic[j, i] * np.dot((xMinusMu).T, (xMinusMu))
            sigmas[i] = (1/m_c[i]) * temp[i]
        #print("sigmas: {}".format(sigmas))
        # compute new likelyhood
        ll_new = 0.0
        for i in range(clusters):
            z = 0
            for j in range(len(dataset)):
                z+= pis[i] * multivariate_normal(mus[i], sigmas[i]).pdf(dataset[j])
            ll_new += np.log(z)
        if abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new
        iterations += 1
    return mus, sigmas
