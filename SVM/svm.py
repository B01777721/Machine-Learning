import numpy as np
from qp import solve_QP
import math

def linear_kernel(xj, xk):
    """
    Kernel Function, linear kernel (ie: regular dot product)

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :return: float32
    """
    #TODO
    l = len(xj)
    s = 0
    for i in range(l):
        s = s + xj[i]*xk[i]
    return s


def rbf_kernel(xj, xk, gamma = 0.1):
    """
    Kernel Function, radial basis function kernel or gaussian kernel

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :param gamma: parameter of the RBF kernel.
    :return: float32
    """
    l = len(xj)
    s = 0
    for i in range(l):
        s = s + pow((xj[i]-xk[i]),2)
    ans = math.exp(-gamma*s)
    return ans
    # TODO

def polynomial_kernel(xj, xk, c = 0, d = 1):
    """
    Kernel Function, polynomial kernel

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :param c: mean of the polynomial kernel (np array)
    :param d: exponent of the polynomial (np array)
    :return: float32
    """
    #TODO
    l = len(xj)
    s = 0
    for i in range(l):
        s = s + xj[i]*xk[i]
    ans = pow((s + c), d)
    return ans

class SVM(object):

    def __init__(self, kernel_func=linear_kernel, lambda_param=.1):
        self.kernel_func = kernel_func
        self.lambda_param = lambda_param
        self.bias = None

    def train(self, inputs, labels):
        """
        train the model with the input data (inputs and labels),
        find the coefficients and constaints for the quadratic program and
        calculate the alphas

        :param inputs: inputs of data, a numpy array
        :param labels: labels of data, a numpy array
        :return: None
        """
        self.train_inputs = inputs #218 x 2 
        self.train_labels = labels #218 x 1 
        Q, c = self._objective_function()
        A, b = self._inequality_constraints()
        E, d = self._equality_constraints()
        # TODO: Uncomment the next line when you have implemented _objective_function(),
        # _inequality_constraints() and _equality_constraints().
        self.alphas = solve_QP(Q, c, A, b, E, d)
        #print(self.alphas.shape)

        #TODO: Given the alphas computed by the quadratic solver, compute the bias
        alpha = self.alphas
        #print(alpha[0])
        Y = self.train_labels
        X = self.train_inputs
        n = len(alpha)
        index = 0
        for i in range(n):
            ai = alpha[i]
            if not np.isclose(ai,0,atol=1e-3) and not np.isclose(ai,1/(2*n*self.lambda_param),atol=1e-3):
            # 0 < alpha[i] and alpha[i] < 1/(2*n*self.lambda_param)
                index = i
                break
        s = 0
        for j in range(n):
            s = s + alpha[j]*(2*Y[j] - 1) * self.kernel_func(X[index],X[j])
        bias = s - (2*Y[index] - 1)

        self.bias = bias


    def _objective_function(self):
        """
        Generate the coefficients on the variables in the objective function for the
        SVM quadratic program.

        Recall the objective function is:
        minimize (1/2)x^T Q x + c^T x

        For specifics on the values for Q and c, see the objective function in the handout.

        :return: two numpy arrays, Q and c which fully specify the objective function.
        """

        #TODO
        X = self.train_inputs
        Y = self.train_labels
        n = len(Y)
        Q = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                Q[i][j] = (2*Y[i]-1)*(2*Y[j]-1)*self.kernel_func(X[j],X[i])
        
        list_ = []
        for k in range(n):
            list_.append(-1)
        c = np.array(list_)
            
        return Q, c

    def _equality_constraints(self):
        """
        Generate the equality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ex = d.

        For specifics on the values for E and d, see the constraints in the handout

        :return: two numpy arrays, E, the coefficients, and d, the values
        """

        #TODO
        X = self.train_inputs
        Y = self.train_labels
        n = len(Y)
        list_ = []
        for i in range(n):
            list_.append(2*Y[i] - 1)
        E = np.array(list_)
        E = E.reshape((1,n))
        d = np.array([0.0])

        return E, d

    def _inequality_constraints(self):
        """
        Generate the inequality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ax <= b.

        For specifics on the values of A and b, see the constraints in the handout

        :return: two numpy arrays, A, the coefficients, and b, the values
        """
        #TODO
        Y = self.train_labels
        n = len(Y)
        m = 2*n
        A = np.zeros((m,n))
        b_list = []
        for i in range(n):
            A[i][i] = 1
            A[i+n][i] = -1


        r = 1.0/(2.0*n*self.lambda_param)

        for i in range(n):
            b_list.append(r)
        for i in range(n):
            b_list.append(0)
        b = np.array(b_list)
        
        return A, b

    def predict(self, input):
        """
        Generate predictions given input.

        :param input: 2D Numpy array. Each row is a vector for which we output a prediction.
        :return: A 1D numpy array of predictions.
        """

        #TODO
        Z = input
        alpha = self.alphas
        X = self.train_inputs
        Y = self.train_labels
        n = len(Y)
        d = len(Z)
        tolist = []
        for k in range(d):
            s = 0
            for i in range(n):
                s = s + alpha[i]*(2*Y[i]-1)*self.kernel_func(X[i],Z[k])
            c = s - self.bias
            if c > 0:
                tolist.append(1)
            else:
                tolist.append(0)
        return np.array(tolist)
        

    def accuracy(self, inputs, labels):
        """
        Calculate the accuracy of the classifer given inputs and their true labels.

        :param inputs: 2D Numpy array which we are testing calculating the accuracy of.
        :param labels: 1D Numpy array with the inputs corresponding true labels.
        :return: A float indicating the accuracy (between 0.0 and 1.0)
        """

        #TODO
        predict_labels = self.predict(inputs)
        true_labels = labels
        total_count = len(labels)
        count = 0
        for i in range(total_count):
            if predict_labels[i] == true_labels[i]:
                count = count + 1
        return count / total_count
        
