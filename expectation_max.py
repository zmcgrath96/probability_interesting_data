import numpy as np
from scipy.stats import multivariate_normal


def em(dataset, clusters, tol=0.01, max_iter=1):
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

    # covariance
    sigmas = [1 for _ in range(clusters)]
    sigmas = np.array(sigmas)


    # likelihood variables
    ll_old = 1
    ll_new = 0.0


    iterations = 0
    while(abs(ll_old - ll_new) > tol or iterations >= max_iter):

        # E Step
        # r is probablity that a point belongs to a cluster
        r_ic = np.zeros((len(dataset), clusters))

        for i in range(len(mus)):
            for j in range(len(dataset)):
                print(pis[i] * multivariate_normal(mus[i], sigmas[i]).pdf(dataset[j]))
                r_ic[i, j] = pis[i] * multivariate_normal(mus[i], sigmas[i]).pdf(dataset[j])

        print(r_ic)

        # M Step

        iterations += 1
