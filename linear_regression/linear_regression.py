import numpy as np

def train(trainX,trainY):
    X = np.ones((trainX.shape[0],trainX.shape[1]+1))
    X[:,1:] = trainX
    #weights = (X^T, X)^(-1) * X^T * Y
    Xt = X.transpose()
    XtX = np.matmul(Xt,X)
    invXtX = np.linalg.inv(XtX)
    weights = np.matmul(np.matmul(invXtX,Xt),trainY)
    return weights

def predict(testX, weights):
    X = np.ones((testX.shape[0],testX.shape[1]+1))
    X[:,1:] = testX
    prediction = np.matmul(X, weights)
    return prediction


def test(testY, prediction):
    m = len(prediction)
    s = 0
    for i in range(m):
        s += abs(prediction[i] - testY[i])
    score = 100*(1-s/m)
    return score
