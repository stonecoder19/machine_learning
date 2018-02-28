import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


plt.ion()
n_observations = 100
fig, ax = plt.subplots(1, 1)
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
ax.scatter(xs, ys)
fig.show()
plt.draw()

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name="weights")
b = tf.Variable(tf.random_normal([1]), name="bias")
Y_pred = tf.add(tf.multiply(X,W),b)

cost = tf.reduce_sum(tf.pow(Y - Y_pred, 2)) / (n_observations -1)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


n_epochs = 1000

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	prev_training_cost = 0.0

	for epoch_i in range(n_epochs):

		writer = tf.summary.FileWriter('./graph1', sess.graph)

		for(x, y) in zip(xs, ys):
			sess.run(optimizer, feed_dict={X: x, Y: y})

		training_cost = sess.run(cost, feed_dict={X: xs, Y:ys})
		print(training_cost)

		if epoch_i % 20 == 0:
			ax.plot(xs, Y_pred.eval(feed_dict={X:xs}, session=sess),'k',alpha= epoch_i / n_epochs)

			fig.show()
			plt.draw()

		if np.abs(prev_training_cost - training_cost) < 0.000001:
			break

		prev_training_cost = training_cost

fig.show()
plt.waitforbuttonpress()