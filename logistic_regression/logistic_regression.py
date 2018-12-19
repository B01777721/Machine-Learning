import numpy as np
import random

def softmax(x):
    e = np.exp(x-np.max(x))
    return e / np.sum(e)


#softmax regression is a generalization of logistic regression
def train(images, labels, numFeatures, numClasses, learning_rate):
    weights = np.zeros((numFeatures, numClasses))

    while True:
        idx = np.arange(len(images))
        np.random.shuffle(idx)
        
        for k in idx:
            l = np.dot(images[k],weights)
            p = softmax(l)

            
            delta = np.zeros(numClasses)
            for i in range(numClasses):
                if i == labels[k]:
                    delta[i] = p[i] - 1
                else:
                    delta[i] = p[i]

            delta_x = np.outer(images[k],delta)
            weights = weights - learning_rate*delta_x

        if np.isclose(delta_x,0,0.001).all():
            break

    return weights

def predict(inputs, weights):
    y = np.zeros(len(inputs))
    for k in range(len(inputs)):
        y_set = np.dot(inputs[k], weights)
        l = np.argmax(y_set)
        y[k] = l
    return y

def test(prediction, y_labels):
    total = len(y_labels)
    count = 0
    for i in range(total):
        if y_labels[i] == prediction[i]:
            count += 1
    return 100* count / total 
