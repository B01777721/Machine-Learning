import numpy as np
import math

class EMNaiveBayesClassifier:

    def __init__(self, num_hidden):
        '''
        @attrs:
            num_hidden  The number of hidden states per class. (An integer)
            priors The estimated prior distribution over the classes, P(Y) (A Numpy Array)
            parameters The estimated parameters of the model. A Python dictionary from class to the parameters
                conditional on that class. More specifically, the dictionary at parameters[Y] should store
                - bjy: b^jy = P(h^j | Y) for each j = 1 ... k
                - bij: b^ij = P(x_i | h^j, Y)) for each i, for each j = 1 ... k
        '''
        self.num_hidden = num_hidden
        self.priors = None
        self.parameters = None
        pass

    def train(self, X, Y, max_iters=10, eps=1e-4):
        '''
            Trains the model using X, Y. More specifically, it learns the parameters of
            the model. It must learn:
                - b^y = P(y) (via MLE)
                - b^jy = P(h^j | Y)  (via EM algorithm)
                - b^ij = P(x_i | h^j, Y) (via EM algorithm)

            Before running the EM algorithm, you should partition the dataset based on the labels Y. Then
            run the EM algorithm on each of the subsets of data.

            :param X 2D Numpy array where each row, X[i, :] contains a sample. Each sample is a binary feature vector.
            :param Y 1D Numpy array where each element Y[i] is the corresponding label for X[i, :]
            :max_iters The maxmium number of iterations to run the EM-algorithm. One
                iteration consists of both the E-step and the M-step.
            :eps Used for convergence test. If the maximum change in any of the parameters is
                eps, then we should stop running the algorithm.
            :return None
        '''
        # TODO

        numInstance = X.shape[0]
        self.priors = np.zeros(2)
        self.parameters= {}
        S0_list = []
        S1_list = []
        for i in range(numInstance):
            if Y[i] == 0:
                S0_list.append(X[i,:])
            else:
                S1_list.append(X[i,:])
        S0 = np.array(S0_list)
        S1 = np.array(S1_list)

        self.priors[0] = len(S0)/numInstance
        self.priors[1] = len(S1)/numInstance

        bij0, bjy0 = self._em_algorithm(S0,self.num_hidden,max_iters,eps)
        bij1, bjy1 = self._em_algorithm(S1,self.num_hidden,max_iters,eps)
        # bij1, bjy1 part should be commented out when trying fake-dataset
        
        self.parameters[0] = {}
        self.parameters[0]["bjy"] = bjy0
        self.parameters[0]["bij"] = bij0
        self.parameters[1] = {}
        self.parameters[1]["bjy"] = bjy1
        self.parameters[1]["bij"] = bij1
        
        #these two lines below replace the two lines above, when trying fake-dataset
        #self.parameters[1]["bjy"] = np.random.random(self.num_hidden)
        #self.parameters[1]["bij"] = np.random.rand(2,self.num_hidden)


        

    def _em_algorithm(self, X, num_hidden, max_iters, eps):
        '''
            EM Algorithm to learn parameters of a Naive Bayes model.

            :param X A 2D Numpy array containing the inputs.
            :max_iters The maxmium number of iterations to run the EM-algorithm. One
                iteration consists of both the E-step and the M-step.
            :eps Used for convergence test. If the maximum change in any of the parameters is
                eps, then we should stop running the algorithm.
            :return the learned parameters as a tuple (b^ij,b^jy)
        '''
        # TODO
        #initialize Q
        self.Q = np.ones((len(X),num_hidden))/num_hidden

        numFeatures = X.shape[1]
        
        t = 0
        bjy = np.random.rand(self.num_hidden)
        bij = np.random.rand(numFeatures,self.num_hidden)
        change1 = float('inf')
        change2 = float('inf')
        
        while t < max_iters:
            print("iteration: " + str(t))
            self.Q = self._e_step(X,num_hidden, bjy, bij)
            new_bij, new_bjy = self._m_step(X, num_hidden, self.Q)
            change1 = np.absolute(np.linalg.norm(bjy) - np.linalg.norm(new_bjy))
            change2 = np.absolute(np.linalg.norm(bij) - np.linalg.norm(new_bij))

            bjy = new_bjy
            bij = new_bij
            t = t + 1
                        
            if change1 <= eps and np.linalg.norm(change2) <= eps:
                break

        return bij, bjy


    def _e_step(self, X, num_hidden, bjy, bij):
        '''
            The E-step of the EM algorithm. Returns Q(t+1) = P(h^j | x, y, theta)
            See the handout for details.

            :param X The inputs to the EM-algorthm. A 2D Numpy array.
            :param num_hidden The number of latent states per class (k in the handout)
            :param bjy at the current iteration (b^jy = P(h^j | y))
            :param bij at the current iteration (b^ij = P(x_i | h^j, y))
            :return Q(t+1)
        '''
        # TODO
        numInstance = X.shape[0]
        numFeatures = X.shape[1]
        
        for i in range(numInstance):
            s= 0
            a = float('-inf')
            
            for j in range(num_hidden):
                p1 = bij[:,j]
                L1 = np.log(np.power(p1,X[i]))
                p2 = 1 - bij[:,j]
                L2 = np.log(np.power(p2,1-X[i]))
                
                z = L1 + L2
                product = np.sum(z) + math.log(bjy[j])
                if product > a:
                    a = product
                self.Q[i,j] = product
              
            s = np.sum(np.exp(self.Q[i]-a))
            s = a + np.log(s)
            self.Q[i] = self.Q[i] - s
            
        self.Q = np.exp(self.Q)

        return self.Q 


    def _m_step(self, X, num_hidden, probs):
        '''
            The M-step of the EM algorithm. Returns the next update to the parameters,
            theta(t+1).

            :param X The inputs to the EM-algorthm. A 2D Numpy array.
            :param num_hidden The number of latent states per class (k in the handout)
            :param probs (Q(t))
            :return theta(t+1) as a tuple (b^ij,b^jy)
        '''
        # TODO
        Sy = X.shape[0]
        numFeatures = X.shape[1]

        Z = np.sum(probs,axis=0)
        bjy = Z/Sy 

        bij = np.zeros((numFeatures,num_hidden))
        for i in range(numFeatures):
            for j in range(num_hidden):
                bij[i,j] = np.dot(probs[:,j],X[:,i]) / Z[j]
                

        return bij, bjy

    def predict(self, X):
        '''
        Returns predictions for the vectors X. For some input vector x,
        the classifier should output y such that y = argmax P(y | x),
        where P(y | x) is approximated using the learned parameters of the model.

        :param X 2D Numpy array. Each row contains an example.
        :return A 1D Numpy array where each element contains a prediction 0 or 1.
        '''
        # TODO

        numInstance = X.shape[0]
        predictions = []
        for i in range(numInstance):
            p0 = self.priors[0]
            s00 = X[i][:,np.newaxis]*self.parameters[0]["bij"]
            s01 = (1-X[i][:,np.newaxis])*(1-self.parameters[0]["bij"])
            s0_ = s00 + s01
            s0 = np.prod(s0_, axis=0)
            M0 = self.parameters[0]["bjy"] * s0
            y0 = p0 * np.sum(M0)

            p1 = self.priors[1]
            s10 = X[i][:,np.newaxis]*self.parameters[1]["bij"]
            s11 = (1-X[i][:,np.newaxis])*(1-self.parameters[1]["bij"])
            s1_ = s10 + s11
            s1 = np.prod(s1_,axis=0)
            M1 = self.parameters[1]["bjy"] * s1
            y1 = p1 * np.sum(M1)

            if y0 >= y1:
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions
        


    def accuracy(self, X, Y):
        '''
            Computes the accuracy of classifier on some data (X, Y). The model
            should already be trained before calling accuracy.

            :param X 2D Numpy array where each row, X[i, :] contains a sample. Each sample is a binary feature vector.
            :param Y 1D Numpy array where each element Y[i] is the corresponding label for X[i, :]
            :return A float between 0-1 indicating the fraction of correct predictions.
        '''
        return np.mean(self.predict(X) == Y)
