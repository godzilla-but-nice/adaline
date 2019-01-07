import numpy as np
import pdb

class AdalineSGD(object):
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
    def __init__(self, eta = 0.01, n_iter = 50, shuffle = True,
                 random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_initialized = False
        self.shuffle = shuffle

    def fit(self, X, Y):
        """
        Fit training data

        Parameters:
        X: training data (array [# of samples x # of features])
        Y: "correct" values (array [# of samples])

        Returns:
        self: updated weights (perceptron object)
        """
        self._initialize_weights(X.shape[1])

        self.cost_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, Y = self._shuffle(X, Y)
            cost = []
            for xi, target in zip(X, Y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(Y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, Y):
        """
        Method for fitting data without initializing weights for online learning
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, Y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, Y)
        return self

    def _initialize_weights(self, size):
        """
        set small initial weights using a normal distribution
        """
        self.rng = np.random.RandomState(self.random_state)
        self.w_ = self.rng.normal(loc = 0.0, scale = 0.01, size=(size + 1))
        self.w_initialized = True

    def _shuffle(self, X, Y):
        """
        Shuffle training data
        """
        r = self.rng.permutation(len(Y))
        return X[r], Y[r]

    def _update_weights(self, xi, target):
        """
        Update the feature weights using the adaline learning rule
        """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[0] += self.eta * error
        self.w_[1:] += self.eta * xi.dot(error)
        cost = 0.5 * error**2
        return cost

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
