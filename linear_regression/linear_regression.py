import numpy as np
from random import shuffle

def train_with_normal(trainX,trainY):
    X = trainX
    #weights = (X^T, X)^(-1) * X^T * Y
    Xt = X.transpose()
    XtX = np.matmul(Xt,X)
    invXtX = np.linalg.inv(XtX)
    weights = np.matmul(np.matmul(invXtX,Xt),trainY)
    return weights

def train_with_gradient(trainX, trainY, learning_rate=0.0001, epochs=500):
    trainX = normalize(trainX)
    m = trainX.shape[0]
    n = trainX.shape[1]
    trainY = y_normalize(trainY)

    #initialize weights
    weights = (np.random.rand(n,)-0.5)/2

    for t in range(epochs):
        #print(t)
        index = [i for i in range(m)]
        shuffle(index)
        for ind in index:
            deriv = np.zeros((n,))
            for j in range(n):
                #print(trainY[ind], np.dot(weights,trainX[ind,:]))
                deriv[j] = 2 * (trainY[ind] - np.dot(weights,trainX[ind,:])) * (-trainX[ind,j])
            weights = weights - learning_rate * deriv
    return weights

def predict(testX, weights):
    prediction = np.matmul(testX, weights)
    return prediction


def test(testY, prediction):
    m = len(prediction)
    s = 0
    for i in range(m):
        s += abs(prediction[i] - testY[i])
    score = 100*(1-s/m)
    return score

def normalize(X):
    m = X.shape[0]
    n = X.shape[1]
    toReturn = np.zeros((m,n))
    for i in range(n):
        mu = np.mean(X[:,i])
        sigma = np.std(X[:,i])
        Xi = (X[:,i] - mu) / sigma
        toReturn[:,i] = Xi
    return toReturn

def y_normalize(Y):
    mu = np.mean(Y)
    sigma = np.std(Y)
    return (Y-mu)/sigma
