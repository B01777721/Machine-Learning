#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains two classifiers: Naive Bayes and Logistic Regression

   Brown CS142, Spring 2018
"""
import random
import numpy as np

class NaiveBayes(object):
    """ Bernoulli Naive Bayes model

    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes classifer with n_classes. """
        self.n_classes = n_classes
        # You are free to add more fields here.
        self.py = np.zeros(10)
        self.pxy = np.zeros((10,784))

    def train(self, data):
        """ Trains the model, using maximum likelihood estimation.

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            None
        """
        # TODO
        images = data[0]
        labels = data[1]
        index, counts = np.unique(labels,return_counts=True)
        #compute py
        for i in index:
            self.py[i] = counts[i]/len(labels)

        #compute pxy
        for k in range(len(images)): #loop through 60000 images
            image = images[k] # for each image
            for d in range(len(image)): #loop through all 784 pixels(features)
                pixel = image[d]
                if pixel == 1: #d = 1~784
                    self.pxy[labels[k]][d] += 1

        for i in range(len(self.pxy)):
            arr = self.pxy[i]
            for j in range(len(arr)):
                d = arr[j]
                prob = d / counts[i]
                if prob == 0:
                    self.pxy[i][j] = 0.01
                elif prob == 1:
                    self.pxy[i][j] = 0.99
                else:
                    self.pxy[i][j] = prob

        
    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """
        #TODO
        #inputs are images
        predicted_labels = np.zeros(len(inputs))
        for index in range(len(inputs)):
            image = inputs[index]
            w = np.copy(self.pxy)
            best_label = np.zeros(self.n_classes)

            for i in range(len(image)):
                if image[i] == 0:
                    w[:,i] = np.subtract(1,self.pxy[:,i])
            
            for p_index in range(len(w)):
                p = w[p_index]
                best_label[p_index] = self.py[p_index] * np.prod(p)
            predicted_labels[index] = np.argmax(best_label)
        return predicted_labels
            

    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        #TODO
        real_labels = data[1]
        predicted_labels = self.predict(data[0])
        l = len(real_labels)
        count = 0
        for i in range(l):
            if real_labels[i] == predicted_labels[i]:
                count += 1
        return count / l
        
        


class LogisticRegression(object):
    """ Multinomial Linear Regression

    @attrs:
        weights: a parameter of the model
        alpha: the step size in gradient descent
        n_features: the number of features
        n_classes: the number of classes
    """
    def __init__(self, n_features, n_classes):
        """ Initializes a LogisticRegression classifer. """
        self.alpha = 0.1  # tune this parameter
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = np.zeros((n_features, n_classes))

    def train(self, data):
        """ Trains the model, using stochastic gradient descent

        @params:
            data: a namedtuple including training data and its information
        @return:
            None
        """
        #TODO
        #f1 = open('test1.csv','w')

        images = data[0]
        labels = data[1]
        while True:
            #shuffle order
            indx = np.arange(len(images))
            np.random.shuffle(indx)
            
            for k in indx:
                l = np.dot(images[k],self.weights)
                p = self._softmax(l)
                delta = np.zeros(self.n_classes)
                for i in range(self.n_classes):
                    if i == labels[k]:
                        delta[i] = p[i] - 1
                    else:
                        delta[i] = p[i]
                delta_x = np.outer(images[k],delta)
                self.weights = self.weights  - self.alpha*delta_x
                #if k % 1000 == 0:
                    #f1.write(str(k)+','+str(1-self.accuracy(data))+',\n')
            if np.isclose(delta_x,0,0.001).all():
                break
        #f1.close()


    def predict(self, inputs):
        """ Compute predictions based on the learned parameters

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            a numpy array of predictions
        """
        #TODO
        # inputs are 10000 x 784
        y = np.zeros(len(inputs))
        for k in range(len(inputs)):
            y_set = np.dot(inputs[k],self.weights)
            l = np.argmax(y_set)
            y[k] = l
        return y
            
            

    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        #TODO
        y_labels = data[1]
        y_predicted = self.predict(data[0])
        total = len(y_labels)
        count = 0
        for i in range(total):
            if y_labels[i] == y_predicted[i]:
                count += 1
        return count / total

    def _softmax(self, x):
        """ apply softmax to an array

        @params:
            x: the original array
        @return:
            an array with softmax applied elementwise.
        """
        e = np.exp(x - np.max(x))
        return e / np.sum(e)
