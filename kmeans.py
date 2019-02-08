import numpy as np
import random as rd
import math

# sqrt sum squares for n dimension
# finds the distance of n dimensional point from another
def distance(a, b):
    dist = 0
    for i in range(len(b)):
        dist += math.pow(b[i] - a[i], 2)

    return math.sqrt(dist)

def closestByIndex(x, y):
    closestCenter = [0 for _ in range(len(x))]
    for i in range(len(x)):
        dist = distance(x[i], y[0])
        closestCenter[i] = 0
        for j in range(1, len(y)):
            newDist = distance(x[i], y[j])
            if newDist < dist:
                dist = newDist
                closestCenter[i] = j

    return closestCenter

# x is data, y are centers
def findCenters(x, y, closest):
    newCenters = [[0 for _ in range(len(x[0]))] for _ in range(len(y))]
    print(closest)
    sums = [[] for _ in range(len(y))]
    for i in range(len(x)):
        sums[closest[i]].append(data[i])
    for i in range(len(y)):
        for j in range(len(x))
            newCenters = np.where(data[j] == y[i])
        newCenters[i] = x[i == closest].mean(0)
        print(.s_newCenters[i])

    newCenters = np.array(newCenters)
    return newCenters

# function to randomize initial centers
def init_centers(data, clusters):
    # tuple with (min, max) for each column in the array
    min_max = [(0, 0) for _ in range(len(data[0]))]
    for i in range(len(min_max)):
        min_max[i] = (np.amin(data, axis=0)[i], np.amax(data, axis=0)[i])

    centers = [[0 for _ in range(len(data[0]))] for _ in range(clusters)]
    for i in range(clusters):
        # for n dimensions, get random number at each dimension
        centers[i] = [rd.uniform(min_max[j][0], min_max[j][1]) for j in range(len(data[0]))]

    return centers

# main function that heads k_means
def k_means(data, clusters):
    centers = init_centers(data, clusters)
    closestIndex = closestByIndex(data, centers)
    newCenters = findCenters(data, centers, closestIndex)
    # if np.all(centers == newCenters):
    #     break
    centers = newCenters
    print(len(centers[0]))
    print(centers)

    return centers, closestIndex
