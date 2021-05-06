import numpy as np


class ridge:
    """ Ridge estimator.
    """

    def __init__(self, lmd=0.):

        self.lmd = lmd
        self.hat = None
        self.hatn = None

    def fit(self, X, y):

        if self.hat is None:
            G = X.T.dot(X) + self.lmd * np.eye(X.shape[1])
            self.hat = np.linalg.solve(G, X.T)

        if self.hatn is None:
            y0 = np.array(list(y[:-1]) + [0])
            self.hatn = self.hat.dot(y0)

        self.beta = self.hatn + y[-1] * self.hat[:, -1]

    def predict(self, x):

        return x.dot(self.beta)

    def conformity(self, y, hat_y):

        return 0.5 * np.square(y - hat_y)
