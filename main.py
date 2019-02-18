import numpy as np
import sys
import kmeans as km
import expectation_max as em
import matplotlib.pyplot as plt
import scipy.stats as ss
import traceback

def main(args):
    discrete_column_number = 0

    # make sure data set passed in
    if not args[1] or 'csv' not in args[1]:
        print('Error: no data csv passed in. Maker sure data is in csv format')
        return
    input_file = args[1]

    data = np.genfromtxt(input_file, delimiter=',', skip_header=1, dtype=float)

    # determine the number of discrete values there are for the number of clusters
    try:
        discrete_column_number = int(args[2])
    except Exception:
        print('Error: no column number with discrete values selected (base 0)')
        return

    discrete_col = np.genfromtxt(input_file, delimiter=',', skip_header=1, usecols=discrete_column_number, dtype=str)
    discrete_vals = {}
    # keep track of row number where each discrete value was found
    for i in range(len(discrete_col)):
        val = discrete_col[i]
        if val not in discrete_vals:
            discrete_vals[val] = []
            discrete_vals[val].append(i)
        else:
            discrete_vals[val].append(i)

    num_discrete = len(discrete_vals)

    # get the two desire columns
    col_one = 0
    col_two = 0
    try:
        col_one = int(args[3])
        col_two = int(args[4]) - 1
    except Exception:
        print("Error: need 2 columns of wanted data (0 based, lower first)")
        return
    data = np.delete(data, np.s_[0:col_one], 1)
    data = np.delete(data, np.s_[1:col_two], 1)
    data = np.delete(data, np.s_[2:len(data)], 1)

    # cluster number AKA number of discrete values
    clusters = num_discrete
    
    # get axis labels
    with open(input_file) as inf:
        line = inf.readline()
    
    line = line.split(',')
    lbl_one = line[col_one].replace('"', '')
    lbl_two = line[col_two].replace('"', '')

    # check for model type
    args_len = len(args)
    if args_len >= 6:
        pdfs = em.em(data, clusters)
        for i in range(len(pdfs)):
            plt.plot(data[:,0], pdfs[i])
        plt.show()
    else:
        centers, index = km.k_means(data, clusters)
        plt.scatter(data[:, 0], data[:, 1], c=index,
            s=50, cmap='viridis')
        plt.xlabel(lbl_one)
        plt.ylabel(lbl_two)
        plt.show()


if __name__ == '__main__':
    main(sys.argv)
