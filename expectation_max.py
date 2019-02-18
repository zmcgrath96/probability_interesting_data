import numpy as np
from scipy.stats import multivariate_normal

def em(dataset, clusters, tol=0.01, max_iter=1000):
    if dataset is None:
        print('No data')
        return
    if clusters is None:
        print('No clusters')
        return

    # change dataset to numpy array for consistency 
    data = np.array(dataset)
    length, width = data.shape

    # weight for each point to each cluster, initialized to random ints
    pis = np.random.randint(0,1,size=(clusters))

    # parameter means, initialized to random integers
    mus = np.random.randint(min(data[:,0]), max(data[:,0]),size=(clusters, width))

    # covariance, initialized to diag of 1s
    sigmas = np.zeros((clusters, width, width))
    for sig in sigmas:
        np.fill_diagonal(sig, 1)
    # likelihood variables
    ll_old = 0.0
    ll_new = 0.0

    iterations = 0
    while(iterations < max_iter):

        # E Step
        # r is probablity that a point belongs to a cluster
        r_ic = np.zeros((length, clusters))

        for i in range(len(mus)):
            for j in range(length):
                print('sigma: {} '.format(sigmas[i]))
                r_ic[i, j] = pis[i] * multivariate_normal(mus[i], sigmas[i]).pdf(data[j])
        # normalize the weights
        r_ic /= r_ic.sum(0)

        # M Step
        # compute news pis
        pis = np.zeros(clusters)
        for i in range(len(mus)):
            for j in range(length):
                pis[i] += r_ic[i, j]
        # normalize to the size of the dataset
        pis /= length

        #compute new mus
        mus = np.zeros((clusters, width))
        for i in range(clusters):
            for j in range(length):
                mus[i] += (r_ic[i, j] * data[j])
            mus[i] = r_ic[i, :].sum()

        #computes new sigmas
        sigmas = np.zeros((clusters, width, width))
        for i in range(clusters):
            for j in range(length):
                xMinusMu = np.reshape(data[i] - mus[i], (2, 1))
                sigmas[i] += r_ic[i, j] * np.dot(xMinusMu, xMinusMu.T)
            sigmas[i] /= r_ic[i,:].sum()

        # compute new likelihood
        ll_new = 0.0
        z = 0
        for i in range(length):
            for j in range(clusters):
                z+= pis[j] * multivariate_normal(mus[j], sigmas[j]).pdf(data[i])
            ll_new += np.log(z)
        print('z: {}'.format(z))
        print('ll_new: {}'.format(ll_new))

        if abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new
        iterations += 1
        print(iterations)
    pdfs = np.empty((clusters, length))
    for i in range(clusters):
        pdfs[i] = multivariate_normal.pdf(data, mean=mus[i], cov=sigmas[i])
    return pdfs
