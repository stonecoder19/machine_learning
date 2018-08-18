from __future__ import division, print_function, unicode_literals

from sklearn.decomposition import PCA


import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings


np.random.seed(42)


plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"


def save_fig(fig_id, tight_layout=True):
	path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
	print("Saving figure", fig_id)
	if tight_layout:
		plt.tight_layout()
	plt.savefig(path, format='png', dpi=300)

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
print(X)

X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)


X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

m, n = X.shape
S =np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)

print(np.allclose(X_centered, U.dot(S).dot(Vt)))


W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

X2D_using_svd = X2D

print(X2D_using_svd)

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)

print(X2D[:5])

# print(X2D_using_svd)
print(X2D_using_svd[:5])

print(np.allclose(X2D, -X2D_using_svd))

X3D_inv = pca.inverse_transform(X2D)

print(np.allclose(X3D_inv, X))

print(np.mean(np.sum(np.square(X3D_inv - X), axis=1)))

X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])

print(np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean_))

print(pca.components_)


print(Vt[:2])

print(pca.explained_variance_ratio_)

print( 1 - pca.explained_variance_ratio_.sum())

print(np.square(s) / np.square(s).sum())






















