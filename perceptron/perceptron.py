import numpy as np


class Perceptron(object):
    """ Perceptron classifier
        Parameters
        eta float learning rate between 0.0 and 1.0
        n_iter int passes over the training dataset.

        Attributes
        w_ ld-array weights after fitting.
        errors_ list number of misclassifications in everty epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, Y):
        """
        Fit trainning data.
        :param x: array-like shape=[n-samples,n-features] Training vectors where n-samples is the number of samples
        :param y: array-like shape=[n_samples]
        :return:
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        """ Calculate net input"""
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        """ Return class label after unit step"""
        return np.where(self.net_input(x) >= 0.0, 1, -1)



    
