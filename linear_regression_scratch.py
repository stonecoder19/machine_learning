import numpy as np
from sklearn.linear_model import LinearRegression


X = np.array([[3],[2],[1]])
Y = np.array([8,6,4])

alpha = 0.1
weights = np.ones((1, 2))

def add_bias(X):
   return np.hstack((np.ones((X.shape[0],1)),X))

def batch_gradient_descent(X,Y,num_iter=1000):
    X = add_bias(X)
    weights = np.zeros(X.shape[1])
    for _ in xrange(num_iter):
        y_hat = np.dot(weights,X.T)

        errors = Y - y_hat

        gradient = np.dot(errors,X)

        weights = weights + alpha * gradient   
        print(weights)
    return weights

def stochastic_gradient_descent(X,Y,num_iter=1000):
    X = add_bias(X)
    weights = np.zeros(X.shape[1])
    for _ in xrange(num_iter):
        for idx,row in enumerate(X):
            y_hat = np.dot(weights,row.T)
    
            error = Y[idx] - y_hat
    
            gradient = error * row
    
            weights = weights + alpha * gradient
            print(weights)
    return weights
          

print("Stochastic gradient descent",stochastic_gradient_descent(X,Y))
print("Batch gradient descent", batch_gradient_descent(X,Y))

model = LinearRegression()
model.fit(X,Y)
print(model.predict(np.array([[5]])))
