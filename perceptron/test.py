import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Perceptron2(object):
    """ Perceptron classifier
        Parameters
        eta float learning rate between 0.0 and 1.0
        n_iter int passes over the training dataset.

        Attributes
        w_ ld-array weights after fitting.
        errors_ list number of misclassifications in everty epoch.

    """

    def __init__(self, eta=0.01, n_tier=10):
        self.eta = eta
        self.n_tier = n_tier

    def fit(self, X, Y):
        """
        Fit trainning data.
        :param x: array-like shape=[n-samples,n-features] Training vectors where n-samples is the number of samples
        :param y: array-like shape=[n_samples]
        :return:
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_tier):
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


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
""" df.tail() """
Y = df.iloc[0:100, 4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

print(X.shape[1], X.shape[0])

"""
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
"""
"""plt.show()"""

"""
ppn = Perceptron2(eta=0.1, n_tier=10)
ppn.fit(X, Y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
"""
