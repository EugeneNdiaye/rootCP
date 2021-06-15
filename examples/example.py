import numpy as np
from sklearn.datasets import make_regression
from rootcp import rootCP, models
from sklearn.linear_model import Lasso

n_samples, n_features = (500, 10)
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=1)
X /= np.linalg.norm(X, axis=0)
y = (y - y.mean()) / y.std()

print("The target is", y[-1])

lmd = 0.5
ridge_regressor = models.ridge(lmd=lmd)
cp = rootCP.conformalset(X, y[:-1], ridge_regressor)
print("Ridge CP set is", cp)


lmd = np.linalg.norm(X.T.dot(y), ord=np.inf) / 30
model = Lasso(alpha=lmd / X.shape[0], warm_start=False)
regression_model = models.regressor(model=model)
cp = rootCP.conformalset(X, y[:-1], regression_model)
print("Lasso CP set is", cp)
