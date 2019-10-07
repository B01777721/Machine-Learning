"""
    This is the class file you will have to fill in.

    You will have to define three classifiers: Nearest Neighbors, K-means and Decision Tree using helper functions
    defined in kneighbors.py, kmeans.py and dtree.py files.

"""

from kmeans import kmeans
from kneighbors import get_neighbors_indices, get_response
from dtree import *

class KNeighborsClassifier(object):
    """
    Classifier implementing the k-nearest neighbors vote.

    @attrs:
        k: The number of neighbors to use, an int
        train_inputs: inputs of training data used to train the model, a 2D Python list
        train_labels: labels of training data used to train the model, a Python list
        n_labels_ : number of labels in the training data, an int,
                    this attribute is used in plot_KNN() to produce classification plot
    """
    def __init__(self, k):
        """
        Initiate K Nearest Neighbors Classifier with some parameters

        :param n_neighbors: number of neighhbors to use, an int
        """
        self.k = k
        self.train_inputs = None
        self.train_labels = None
        self.n_labels_ = None

    def train(self, X, y):
        """
        train the data (X and y) to model, calculate the number of unique labels and store it in self.n_labels_

        :param X: inputs of data, a 2D Python list
        :param y: labels of data, a Python list
        :return: None
        """
        self.train_inputs = X
        self.train_labels = y
        self.n_labels_ = len(np.unique(y))

    def predict(self, X):
        """
        Compute predictions of input X

        :param X: inputs of data, a 2D Python list
        :return: a Numpy array of predictions
        """

        # For each data point in X:
        # 1. Compute the k nearest neighbors (indices)
        # 2. Compute the highest label response given the k nearest neighbors
        # Use the helper methods!

        # TODO

        # for each x in X, we find k nearest objects in train_inputs
        # and have them vote to find the majority label

        lst = []
        for x in X:
            k_neighbors_indices = get_neighbors_indices(self.train_inputs,x,self.k)
            l = get_response(self.train_labels,k_neighbors_indices)
            lst.append(l)
        arr = np.array(lst,dtype=object)
        return arr


    def accuracy(self, data):
        """
        Compute accuracy of the model when applied to data

        :param data: a namedtuple including inputs and labels
        :return: a float number indicating accuracy
        """
        # TODO: Compute the portion of data with correctly predicted labels
        label1 = self.predict(data.inputs)
        label2 = data.labels
        l = len(label1)
        count = 0
        for i in range(l):
            if label1[i] == label2[i]:
                count += 1
        return count / l 
        

class KmeansClassifier(object):
    """
    K-means Classifier via Iterative Improvement

    @attrs:
        k: The number of clusters to form as well as the number of centroids to generate (default = 3), an int
        tol: Relative tolerance with regards to inertia to declare convergence, a float number,
                the default value is set to 0.0001
        max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int,
                  the default value is set to 500 iterations
        cluster_centers_: a Python dictionary where each (key, value) pair is
                            (a class label[an int], k cluster centers of that class label[a Numpy array])

    K-means is not a classification algorithm, it is an unsupervised learning algorithm. You will be creating K
    cluster centers for EACH label (k * #labels total). The label of the closest center is then used to classify data.

    """

    def __init__(self, n_clusters = 3, max_iter = 500, threshold = 1e-4):
        """
        Initiate K-means with some parameters
        """
        self.k = n_clusters
        self.tol = threshold
        self.max_iter = max_iter
        self.cluster_centers_ = dict()

    def train(self, X, y):
        """
        Compute K-means clustering over data with each class label and store your result in self.cluster_centers_
        You should use kmeans helper function from kmeans.py

        :param X: inputs of training data, a 2D Python list
        :param y: labels of training data, a Python list
        :return: None
        """
        # TODO
        y_arr = np.array(y)
        labels = np.unique(y_arr)
        for l in range(len(labels)):
            y_ = labels[l]
            Xy = []
            for i in range(len(X)):
                if y[i] == y_:
                    Xy.append(X[i])
            k_clusters = kmeans(Xy,self.k,self.max_iter,self.tol)
            self.cluster_centers_[y_] = k_clusters
        
    def sq_distance(self,xlist,xarr):
        # computing squared distance between xlist and xarr
        l = len(xlist)
        s = 0 
        for i in range(l):
            s += pow(xlist[i] - xarr[i],2)
        return s 

    def predict(self, X):
        """
        Predict the label of each sample in X, which is the label of the closest cluster center each sample in X belongs to
        Be sure to identify the closest center out of all (k * #labels) centers

        :param X: inputs of data, a 2D Python list
        :return: a Python list of labels predicted by model
        """
        # TODO
        predict_list = []
        for i in range(len(X)):
            x = X[i] # for each x-instance
            lst_ = []
            for y in self.cluster_centers_:
                centroids = self.cluster_centers_[y]
                lst = []
                for j in range(len(centroids)):
                    r = self.sq_distance(x,centroids[j])
                    lst.append(r)
                min_val = min(lst)
                lst_.append([min_val,y])
            lst_.sort()
            best_y = lst_[0][1]
            predict_list.append(best_y)
        return predict_list
            

    def accuracy(self, data):
        """
        Compute accuracy of the model when applied to data

        :param data: a namedtuple including inputs and labels
        :return: a float number indicating accuracy
        """
        # TODO: Compute the portion of data with correctly predicted labels
        labels1 = self.predict(data.inputs)
        labels2 = data.labels
        l = len(labels1)
        count = 0 
        for i in range(l):
            if labels1[i] == labels2[i]:
                count += 1
        return count / l 

class DecisionTree:
    """
    A DecisionTree with ranges. Can handle (multi-class) classification tasks
    with continious inputs.

    @attrs:
        data: The data that will be used to construct the decision tree, as a python list of lists.
        gain_function: The gain_function specified by the user.
        max_depth: The maximum depth of the tree, a int.
    """

    def __init__(self, data, validation_data=None,  gain_function='entropy', max_depth=40):
        """
        Initiate the deccision tree with some parameters
        """
        self.max_depth = max_depth
        self.root = Node()

        if gain_function=='entropy':
            self.gain_function = entropy
        elif gain_function=='gini_index':
            self.gain_function = gini_index
        else:
            print("ERROR: GAIN FUNCTION NOT IMPLEMENTED")

        indices = list(range(1, len(data[0])))
        split_recurs(self.root, data, indices, self.gain_function, self.max_depth)

        if not (validation_data is None):
            self._prune_recurs(self.root, validation_data)

    def predict(self, X):
        """
        Predicts the label of each sample in X

        :param X: A dataset as a python list of lists
        :return: A list of labels predicted by the trained decision tree
        """
        labels = [predict_recurs(self.root, data) for data in X]
        return labels

    def accuracy(self, data):
        """
        Computes accuracy of the model when applied to data

        :param data: dataset with the first column as the label, a python list of lists.
        :return: A float indicating accuracy (between 0 and 1)
        """
        cnt = 0.0
        test_Y = [row[0] for row in data]
        pred =  self.predict(data)
        for i in range(0, len(test_Y)):
            if test_Y[i] == pred[i]:
                cnt+=1
        return float(cnt/len(data))

    def print_tree(self):
        """
        Visualize the decision tree
        """
        print_tree(self.root)
