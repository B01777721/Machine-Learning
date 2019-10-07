"""
    This is a file you will have to fill in.

    It contains helper functions required by K Nearest Neighbors method
    Note: get_neighbors_indices and get_response are designed to be called in sequence by models.py
"""
import numpy as np
import operator
import math

def euclidean_distance(input1, input2):
    """
    Compute the euclidean distance between input1 and input2

    :param input1: input data point 1, a Python list
    :param input2: input data point 2, a Python list
    :return: euclidean distance between input1 and input2, a float number
    """
    # TODO

    # Recall the definition of Euclidean distance for two vectors x,y
    # d = sqrt(sum(x_i - y_i)^2)
    l = len(input1)
    s = 0
    for i in range(l):
        s += pow((input1[i] - input2[i]),2)
    d = math.sqrt(s)
    return d

def get_neighbors_indices(training_inputs, test_instance, k):
    """
    Get the indices of k closest neighbors to testInstance using Euclidean Distance
    Use the euclidean_distance helper function

    :param training_inputs: inputs of training data, a 2D Python list
    :param test_instance: an instance input of test data, a Python list
    :param k: number of neighbors used, an int
    :return: a Python list of indices of k closest neighbors from the training set for a given test instance
    """
    # TODO
    train_len = len(training_inputs)
    lst = []
    for j in range(train_len):
        d = euclidean_distance(training_inputs[j],test_instance)
        lst.append([d,j])
    lst.sort()
    k_labels_indices = []
    for i in range(k):
        k_labels_indices.append(lst[i][1])
    return k_labels_indices
    
    
def get_response(training_labels, neighbor_indices):
    """
    Get the most commonly voted response from a number of neighbors

    :param training_labels: labels of training data, a Python list
    :param neighbor_indices: a Python list of indices of k closest neighbors from the training data
    :return: the class/label with the highest vote, an int
    """
    # TODO
    label_list = []
    for i in neighbor_indices:
        label_list.append(training_labels[i])
    return max(set(label_list),key=label_list.count)
