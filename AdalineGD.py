import numpy as np
import pdb

class AdalineGD(object):
    """
    Adaptive linear neuron classifier

    Parameters:
    eta: learning rate (float)
    n_iter: number of training passes (int)
    random_state: seed for weight initialization (int)

    Attributes:
    w_: weights after fitting (1D array [# of features])
    cost_: sum of squared error in classification in each epoch
    """
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, Y):
        """
        Fit training data

        Parameters:
        X: training data (array [# of samples x # of features])
        Y: "correct" values (array [# of samples])

        Returns:
        self: updated weights (perceptron object)
        """
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc = 0.0, scale = 0.01, size=(X.shape[1] + 1))

        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (Y - output)
            self.w_[0] += self.eta * errors.sum()
            self.w_[1:] += self.eta * X.T.dot(errors)
            cost = 0.5 * (errors**2).sum()
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """
        net input from a sample
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """
        Linear activation function
        """
        return X

    def predict(self, X):
        """
        return label using classifier step function
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
