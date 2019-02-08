import numpy as np
import sys
import kmeans as km
import matplotlib.pyplot as plt

def main(args):
    # make sure data set passed in
    if not args[1] or 'csv' not in args[1]:
        print('Error: no data csv passed in. Maker sure data is in csv format')
    input_file = args[1]
    data = np.genfromtxt(input_file, delimiter=',', skip_header=1, dtype=float)

    # TODO: parameterize later
    data = np.delete(data, np.s_[0:2], 1)
    data = np.delete(data, np.s_[1:5], 1)
    data = np.delete(data, np.s_[2:len(data)], 1)

    # cluster number for k_means
    clusters = 2

    # check for model type
    if not args[2] or not 'em' in args[2]:
        if args[2] and int(args[2]):
            clusters = int(args[2])
        centers, index = km.k_means(data, clusters)
        plt.scatter(data[:, 0], data[:, 1], c=index,
            s=50, cmap='viridis');
        plt.show();


if __name__ == '__main__':
    main(sys.argv)
