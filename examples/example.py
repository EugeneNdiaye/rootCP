import numpy as np
from sklearn.datasets import make_regression
from rootcp import rootCP, models
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit

n_samples, n_features = (500, 10)
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=1)
X /= np.linalg.norm(X, axis=0)
y = (y - y.mean()) / y.std()
lmd = 0.5

# ridge_regressor = models.ridge(lmd=lmd)
# cp = rootCP.conformalset(X, y[:-1], ridge_regressor)
# print("CP set is", cp)

method = "GradientBoosting"
method = "MLP"
# method = "AdaBoost"
# method = "RandomForest"
# method = "OMP"
# method = "Lasso"

if method == "GradientBoosting":
    model = GradientBoostingRegressor(warm_start=True)

if method == "MLP":
    model = MLPRegressor(warm_start=False)

if method == "AdaBoost":
    model = AdaBoostRegressor(warm_start=True)

if method == "RandomForest":
    # For randomForest I dont know yet if it is safe to use warm_start
    model = RandomForestRegressor(warm_start=False)

if method == "OMP":
    # Do not have a warm_start
    tol_omp = 1e-2 * np.linalg.norm(y[:-1]) ** 2
    model = OrthogonalMatchingPursuit(tol=tol_omp)

if method == "Lasso":
    model = Lasso(alpha=lmd / X.shape[0], warm_start=False)

regression_model = models.regressor(model=model)
cp = rootCP.conformalset(X, y[:-1], regression_model)
print(method, "CP set is", cp)
