import numpy as np
import sys
import kmeans as km
import matplotlib.pyplot as plt

def main(args):
    discrete_column_number = 0

    # make sure data set passed in
    if not args[1] or 'csv' not in args[1]:
        print('Error: no data csv passed in. Maker sure data is in csv format')
    input_file = args[1]
    def convertDiag(x):
        if x == 'B':
            return 0.0
        else:
            return 1.0
    data = np.genfromtxt(input_file, delimiter=',', skip_header=1, dtype=float, converters={1: convertDiag})

    # determine the number of discrete values there are for the number of clusters
    if args[2]:
        discrete_column_number = int(args[2])

    discrete_col = np.genfromtxt(input_file, delimiter=',', skip_header=1, usecols=discrete_column_number, dtype=str)
    discrete_vals = {}
    for a in discrete_col:
        if a not in discrete_vals:
            discrete_vals[a] = 1

    num_discrete = len(discrete_vals)

    # TODO: parameterize later
    data = np.delete(data, np.s_[0:2], 1)
    data = np.delete(data, np.s_[1:5], 1)
    data = np.delete(data, np.s_[2:len(data)], 1)

    # cluster number AKA number of discrete values
    clusters = num_discrete

    # check for model type
    if not args[2] or not 'em' in args[2]:
        if args[2] and int(args[2]):
            clusters = int(args[2])
        centers, index = km.k_means(data, clusters)
        plt.scatter(data[:, 0], data[:, 1], c=index,
            s=50, cmap='viridis')
        plt.show()


if __name__ == '__main__':
    main(sys.argv)
