"""
    This is a file you will have to fill in.

    It contains helper functions required by K-means method via iterative improvement

"""
import numpy as np
import random
import math

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids

    :param k: number of cluster centroids, an int
    :param inputs: a 2D Python list, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    lst = random.sample(inputs,k)
    arr = np.array(lst)
    return arr
    


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance

    :param inputs: inputs of data, a 2D Python list
    :param centroids: a Numpy array of k current centroids
    :return: a Python list of centroid indices, one for each row of the inputs
    """
    # TODO
    indices = []
    for i in range(len(inputs)):
        x = inputs[i]
        lst = []
        for j in range(len(centroids)):
            s = 0
            for l in range(len(x)):
                s += pow(x[l] - centroids[j][l],2)
            d = math.sqrt(s)
            lst.append([d,j])
        lst.sort()
        c = lst[0][1]
        indices.append(c)
    return indices

# helper function for update_step 
def sum_and_average(lst,m):
    # lst is 2d python list
    l = len(lst[0])
    return_list = []
    for j in range(l):
        s = 0 
        for i in range(len(lst)):
            s += lst[i][j]
        return_list.append(s/m)
    return return_list

def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster - the average of all data points in the cluster

    :param inputs: inputs of data, a 2D Python list
    :param indices: a Python list of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    centroids = []
    for j in range(k): # for each centeroids
        m = 0
        lst = []
        for i in range(len(inputs)):
            if indices[i] == j:
                m += 1
                lst.append(inputs[i])
        jth_center = sum_and_average(lst,m)
        centroids.append(jth_center)
    return np.array(centroids)

        
        
def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    Use init_centroids, assign_step, and update_step!
    The only computation that should occur within this function is checking 
    for convergence - everything else should be handled by helpers

    :param inputs: inputs of data, a 2D Python list
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: relative tolerance with regards to inertia to declare convergence, a float number
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO

    #initialize centroids
    centroids = init_centroids(k,inputs)

    i = 0
    converged = False
    while i <= max_iter and converged == False:
        prev_centroids = centroids
        indices = assign_step(inputs,prev_centroids)
        centroids = update_step(inputs,indices,k)
        i += 1
        temp = True
        for j in range(len(centroids)):
            if sum(abs(centroids[j] - prev_centroids[j]))/sum(prev_centroids[j]) > tol:
                temp = False
        converged = temp
    return centroids
        

        
        
