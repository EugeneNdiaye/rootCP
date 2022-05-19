from generalized_z_path import assymetric_grad, assymetric_loss, inv_assymetric_grad, inv_robust_grad
from cp_tool import cpregressor
import numpy as np
from generalized_z_path import FenCon, lasso_loss, lasso_gradient, inv_lasso_gradient, assymetric_loss, assymetric_grad, inv_assymetric_grad, robust_loss, robust_grad, inv_robust_grad

class lassocpregressor(cpregressor):
	def __init__(self, lmd):
		super().__init__(lasso_loss, lasso_gradient, inv_lasso_gradient, lmd)

class robustcpregressor(cpregressor):
	def __init__(self, lmd):
		super().__init__(robust_loss, robust_grad, inv_robust_grad, lmd)

class asymetriccpregressor(cpregressor):
	def __init__(self, lmd):
		super().__init__(assymetric_loss, assymetric_grad, inv_assymetric_grad, lmd)	

class cpregressor():
	def __init__(self, loss, grad, invgradf, lmd):
		self.invgradf = invgradf
		self.loss = loss
		self.grad = grad
		self.lmd = lmd

	def fit(self, X, y):
		m = FenCon(self.loss, self.grad, self.invgradf)
		betas, cands = m.solve(X, y[:-1])


		z = y[-1]
		if z > cands[0]:
			currb = betas[0] 
		elif z < cands[-1]:
			currb = betas[-1]
		else:
			for idx, cand in enumerate(cands):
				
				if cand > z:
					currb = betas[idx]
					break
	
		
		self.aset = np.argwhere(currb != 0)
		pseudo_temp = np.linalg.pinv(X[:, self.aset] @ X[:, self.aset].T)
		Pi_A = (pseudo_temp @ X[:, self.aset]).dot(self.eta[self.aset])
		Pi_At = np.linalg.solve(X[:, self.aset].T @ X[:, self.aset], X[:, self.aset].T)
		v_a = np.sign(currb)

		self.beta = Pi_A @ self.invgradf(-self.lmd * Pi_At * v_a)

	def predict(self, X):
		return X.dot(self.beta)
		
	def conformity(self, y, y_pred):
		return 0.5 * np.square(y - y_pred)



class ridge:
	""" Ridge estimator.
	"""

	def __init__(self, lmd=0.1):

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

	def predict(self, X):

		return X.dot(self.beta)

	def conformity(self, y, y_pred):

		return 0.5 * np.square(y - y_pred)


class regressor:

	def __init__(self, model=None, s_eps=0., conform=None):

		self.model = model
		self.coefs = []
		self.s_eps = s_eps
		self.conform = conform

	def fit(self, X, y):

		refit = True

		for t in range(len(self.coefs)):

			if self.s_eps == 0:
				break

			if abs(self.coefs[t][0] - y[-1]) <= self.s_eps:
				self.beta = self.coefs[t][1].copy()
				refit = False
				break

		if refit:
			self.beta = self.model.fit(X, y)
			if self.s_eps != 0:
				self.coefs += [[y[-1], self.beta.copy()]]

	def predict(self, X):

		if len(X.shape) == 1:
			X = X.reshape(1, -1)

		return self.model.predict(X)

	def conformity(self, y, y_pred):

		if self.conform is None:
			return np.abs(y - y_pred)

		else:
			return self.conform(y, y_pred)
