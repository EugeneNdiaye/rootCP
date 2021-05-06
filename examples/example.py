import numpy as np
from sklearn.datasets import make_regression
from rootcp import rootCP, models

n_samples, n_features = (300, 50)
X, y = make_regression(n_samples=n_samples, n_features=n_features)
X /= np.linalg.norm(X, axis=0)
y = (y - y.mean()) / y.std()
lmd = 0.5

ridge_regressor = models.ridge(lmd=lmd)
cp = rootCP.conformalset(X, y[:-1], ridge_regressor)
print("CP set is", cp)
