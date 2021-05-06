import numpy as np
from scipy.optimize import root_scalar
from scipy.special import expit


def rank(u, gamma=None):

    if gamma is None:
        return np.sum(u - u[-1] <= 0)

    sigmoid = 1 - expit(gamma * (u - u[-1]))
    return np.sum(sigmoid)


def conformalset(X, y, model, alpha=0.1, gamma=None, tol=1e-4):
    """ Compute full conformal prediction set with root-finding solver.

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape = (n_samples - 1,)
        Target values.
    model : class that represents a regressor with a model.fit,
            model.predict and model.conformity.
    alpha : float in (0, 1)
            Coverage level.
    gamma : float
            Smoothing parameter.
    tol : float
          Tolerance error for the root-finding solver.
    Returns
    -------
    list : list of size 2
           conformal set [l(alpha), u(alpha)].
    """

    def pvalue(z):

        yz[-1] = z
        model.fit(X, yz)
        scores = model.conformity(yz, model.predict(X))

        return 1 - rank(scores, gamma) / X.shape[0]

    def objective(z):
        # we use a bit more conservative alpha to include all roots when
        # the p_value function is piecewise constant.
        return pvalue(z) - (alpha - 1e-11)

    # TODO: test Initial condition
    z_min, z_max = np.min(y), np.max(y)
    yz = np.array(list(y) + [0])
    model.fit(X, yz)
    z_0 = model.predict(X[-1])

    # root-finding
    algo = "bisect" if gamma is None else "brenth"
    left = root_scalar(objective, bracket=[z_min, z_0], method=algo, xtol=tol)
    right = root_scalar(objective, bracket=[z_0, z_max], method=algo, xtol=tol)

    return [left.root, right.root]
