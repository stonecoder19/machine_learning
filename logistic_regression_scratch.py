import numpy as np
import matplotlib.pyplot as plt


X1 = np.random.multivariate_normal([0,0],[[1, .75], [.75, 1]], 5000)
X2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], 5000)


features = np.vstack((X1,X2)).astype(np.float32)
labels = np.hstack((np.zeros(5000),np.ones(5000)))

alpha = 0.1
#print(features)
#print(add_bias(features))

def add_bias(X):
	return np.hstack((np.ones((X.shape[0],1)),X))

#print(add_bias(features))

def sigmoid(X):
	return 1 / (1 + np.exp(-X))


def sgd_logistic_regression(X, Y,num_iter=1000):
	X = add_bias(X)
	weights = np.zeros(X.shape[1])

	for _ in xrange(num_iter):

		for idx,row in enumerate(X):
			yhat = np.dot(weights,row.T)

			yhat = sigmoid(yhat)
			
			error = Y[idx] - yhat
			
			gradient = error * row
			
			weights = weights + alpha * gradient
                        print(weights)

	return weights


weights = sgd_logistic_regression(features, labels)

