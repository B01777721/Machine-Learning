import numpy as np
import random

def l2_loss(predictions,Y):
    '''
        Computes L2 loss (sum squared loss) between true values, Y, and predictions.

        :param Y A 1D Numpy array with real values (float64)
        :param predictions A 1D Numpy array of the same size of Y
        :return L2 loss using predictions for Y.
    '''
    # TODO
    m = len(predictions)
    s = 0
    for i in range(m):
        s = s + pow((Y[i]-predictions[i]),2)
    return s

def sigmoid(x):
    '''
        Sigmoid function f(x) =  1/(1 + exp(-x))
        :param x A scalar or Numpy array
        :return Sigmoid function evaluated at x (applied element-wise if it is an array)
    '''
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0)))

def sigmoid_derivative(x):
    '''
        First derivative of the sigmoid function with respect to x.
        :param x A scalar or Numpy array
        :return Derivative of sigmoid evaluated at x (applied element-wise if it is an array)
    '''
    # TODO
    # derivative of sigmoid: d(sig(x))/dx = sig(x) * 1-sig(x)
    return sigmoid(x)*(1-sigmoid(x))


class LinearRegression:
    '''
        LinearRegression model that minimizes squared error using matrix inversion.
    '''
    def __init__(self):
        '''
        @attrs:
            weights The weights of the linear regression model.
        '''
        self.weights = None

    def train(self, X, Y):
        '''
        Trains the LinearRegression model by finding the optimal set of weights
        using matrix inversion.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return None
        '''
        # TODO
        m = len(X) # 3918
        n = len(X[0]) 

        A = np.zeros((n,n))
        b = np.zeros((n,))
        for i in range(m):
            Xi = X[i]
            Xi_T = Xi.transpose()
            A = A + np.matmul(Xi[:,None],Xi_T[None,:])
            b = b + np.dot(Xi,Y[i])

        w = np.matmul(np.linalg.inv(A),b)
        self.weights = w

            

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        # TODO
        w = self.weights
        m = len(X)
        h = []
        for i in range(m):
            s = np.dot(w,X[i])
            h.append(s)
        return np.array(h)

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]

class OneLayerNN:
    '''
        One layer neural network trained with Stocastic Gradient Descent (SGD)
    '''
    def __init__(self):
        '''
        @attrs:
            weights The weights of the neural network model.
        '''
        self.weights = None
        pass

    def train(self, X, Y, learning_rate=0.001, epochs=250, print_loss=True):
        '''
        Trains the OneLayerNN model using SGD.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :param learning_rate The learning rate to use for SGD
        :param epochs The number of times to pass through the dataset
        :param print_loss If True, print the loss after each epoch.
        :return None
        '''
        # TODO
        m = len(X)
        n = len(X[0])
        # initialize weights
        w = (np.random.rand(n,) - 0.5) / 2
        
        for t in range(epochs):
            # for each epoch
            index = [i for i in range(m)]
            random.shuffle(index)

            for ind in index:
                deriv = np.zeros((n,))
                for j in range(n):
                    deriv[j] = 2 * (Y[ind] - np.dot(w,X[ind])) * (-X[ind][j])
                w = w - learning_rate * deriv 
           
        self.weights = w


    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        #TODO
        w = self.weights
        m = len(X)
        h = []
        for i in range(m):
            s = np.dot(w,X[i])
            h.append(s)
        return np.array(h)

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]


class TwoLayerNN:

    def __init__(self, hidden_size, activation=sigmoid, activation_derivative=sigmoid_derivative):
        '''
        @attrs:
            activation: the activation function applied after the first layer
            activation_derivative: the derivative of the activation function. Used for training.
            hidden_size: The hidden size of the network (an integer)
            output_neurons: The number of outputs of the network
        '''
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size

        # In this assignment, we will only use output_neurons = 1.
        self.output_neurons = 1

        # These are the learned parameters for the 2-Layer NN you will implement
        self.hidden_weights = None
        self.hidden_bias = None
        self.output_weights = None
        self.output_bias = None

    def train(self, X, Y, learning_rate=0.01, epochs=1000, print_loss=True):
        '''
        Trains the TwoLayerNN with SGD using Backpropagation.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :param learning_rate The learning rate to use for SGD
        :param epochs The number of times to pass through the dataset
        :param print_loss If True, print the loss after each epoch.
        :return None
        '''
        #TODO
        n = len(X)
        d = len(X[0])
        m = self.hidden_size

        w = (np.random.rand(d, m) - 0.5)/4
        b1 = (np.random.rand(m,) - 0.5)/4
        v = (np.random.rand(m,) - 0.5)/4
        b2 = (random.random() - 0.5)/4

        for t in range(epochs):
            index = [i for i in range(n)]
            random.shuffle(index)
            for ind in index:
                h = self.activation(np.matmul(X[ind], w) + b1)
                z = np.dot(h,v) + b2
                deriv = self.activation_derivative(h)

                b2 = b2 - learning_rate*2*(z-Y[ind])
                v = v - learning_rate*2*(z-Y[ind])*h
                b1 = b1 - learning_rate*2*(z-Y[ind])*np.multiply(v, deriv)
                for i in range(m):
                    C = 2*(z-Y[ind])*v[i]*self.activation_derivative(np.dot(w[:,i], X[ind]) + b1[i])
                    for j in range(d):
                        w[j][i] = w[j][i] - learning_rate*C*X[ind][j]

            self.hidden_weights = w
            self.hidden_bias = b1
            self.output_weights = v
            self.output_bias = b2

            if print_loss is True:
                print(self.loss(X,Y))


    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        #TODO
        n = len(X)
        pred = []
        for i in range(n):
            h = self.activation(np.matmul(X[i], self.hidden_weights) + self.hidden_bias)
            z = np.dot(h,self.output_weights) + self.output_bias
            pred.append(z)

        return np.array(pred)

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]

# extra credit functions (commented)
#def linear(x):
#    return x

#def linear_derivative(x):
#    if hasattr(x,"__len__"):
#        l = len(x)
#        return np.ones(l,)
#    else:
#        return 1

#def step(x):
#    return np.where(x > 0, 1)

#def step_derivative(x):
#    if hasattr(x,"__len__"):
#        l = len(x)
#        np.zeros(l,)
#    else:
#        return 0

#def relu(x):
#    return np.where(x > 0, x, 0)

#def relu_derivative(x):
#    return np.where(x > 0, 1, 0)

