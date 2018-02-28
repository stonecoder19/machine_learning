
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('MNIST-data/', one_hot=True)

print(mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)

print(mnist.train.images.shape, mnist.train.labels.shape)

plt.imshow(np.reshape(mnist.train.images[100,:], (28, 28)), cmap='gray')
plt.show()

n_input = 784
n_output = 10
net_input = tf.placeholder(tf.float32, [None, n_input])


W = tf.Variable(tf.zeros([n_input,n_output]))
b = tf.Variable(tf.zeros([n_output]))
net_output = tf.nn.softmax(tf.matmul(net_input,W)+ b)

y_true = tf.placeholder(tf.float32,[None,n_output])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=net_output)

correct_prediction = tf.equal(tf.argmax(y_true,1), tf.argmax(net_output,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float32"))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

batch_size = 100
n_epoch = 50

for epoch_i in range(n_epoch):
	for i in range(mnist.train.num_examples // batch_size):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		sess.run(optimizer,feed_dict={net_input:batch_xs,y_true:batch_ys})

	print(sess.run(accuracy, feed_dict={net_input: mnist.validation.images, y_true: mnist.validation.labels}))

print (sess.run(accuracy, feed_dict={net_input:mnist.test.images, y_true:mnist.test.labels}))