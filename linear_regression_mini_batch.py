import numpy as np
import tensorflow as tf


datapoint_size = 1000
batch_size = 1
steps = 10000
actual_W = 2
actual_b = 10
learn_rate = 0.001
log_file = "/tmp/feature_1_batch_1"


x = tf.placeholder(tf.float32, [None, 1], name="x")
W = tf.Variable(tf.zeros([1,1]), name="W")
b = tf.Variable(tf.zeros([1]),name="b")

with tf.name_scope("Wx_b") as scope:
	product = tf.matmul(x,W)
	y = product + b


W_hist = tf.summary.histogram("weights", W)
b_hist = tf.summary.histogram("biases", b)
y_hist = tf.summary.histogram("y", y)

y_ = tf.placeholder(tf.float32,[None, 1], name="y_")

with tf.name_scope("cost") as scope:
	cost = tf.reduce_mean(tf.square(y_-y))
	cost_sum = tf.summary.scalar("cost",cost)

with tf.name_scope("train") as scope:
	train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

all_xs = []
all_ys = []

for i in range(datapoint_size):

	all_xs.append(i%10)
	all_ys.append(actual_W*(i%10)+actual_b)

all_xs = np.transpose([all_xs])
all_ys = np.transpose([all_ys])

sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_file, sess.graph_def)

init = tf.initialize_all_variables()
sess.run(init)

for i in range(steps):
	if datapoint_size == batch_size:
		batch_start_idx = 0
	elif datapoint_size < batch_size:
		raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" %(datapoint_size, batch_size))
	else:
		batch_start_idx = (i*batch_size) % (datapoint_size-batch_size)

	batch_end_idx = batch_start_idx+ batch_size
	batch_xs = all_xs[batch_start_idx:batch_end_idx]
	batch_ys = all_ys[batch_start_idx:batch_end_idx]
	xs = np.array(batch_xs)
	ys = np.array(batch_ys)

	if i % 10 == 0:
		all_feed = {x:all_xs, y_: all_ys}
		result = sess.run(merged,feed_dict=all_feed)
		writer.add_summary(result, i)
	else:
		feed = {x: xs, y_: ys}
		sess.run(train_step, feed_dict=feed)
		print("y: %s" % sess.run(y, feed_dict=feed))
		print("y_: %s" %ys)
		print("cost: %f" %sess.run(cost, feed_dict=feed))
		print("After %d iteration:" %i)
		print("W: %f" % sess.run(W))
		print("b: %f" %sess.run(b))