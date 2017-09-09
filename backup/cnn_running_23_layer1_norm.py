import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
#import utils
import logging
import dataset
import os

sess = tf.InteractiveSession()
ds = dataset.read_data_sets('/home/nikhil/data/',one_hot=True)
print(ds.train.images.shape)
print(ds.train.labels.shape)
print(ds.test.images.shape)
print(ds.test.labels.shape)
x = tf.placeholder(tf.float32, shape=[None, 3584])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
phase_train = tf.placeholder(tf.bool , name = 'phase_train')
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def batch_norm(x , n_out , phase_train , scope = 'bn'):
	
	with tf.variable_scope(scope):
		#pred = tf.placeholder(tf.bool , shape = [])
		#pred = True
		beta = tf.Variable(tf.constant(0.0 , shape = [n_out]) , name = 'beta' , trainable = True)
		gamma = tf.Variable(tf.constant(1.0 , shape=[n_out]) , name = 'gamma' , trainable = True)
		batch_mean , batch_var = tf.nn.moments(x , [0,1,2] ,name = 'moments')
		ema = tf.train.ExponentialMovingAverage(decay = 0.5)
		
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean , batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean) , tf.identity(batch_var)
		mean , var = tf.cond(phase_train , mean_var_with_update, lambda: (ema.average(batch_mean) , ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean , var , beta, gamma, 1e-3)
	return normed
### First Layer #######
W_conv1 = weight_variable([5, 5, 1, 10])
b_conv1 = bias_variable([10])

x_image = tf.reshape(x, [-1,56,64,1])

h_conv1 = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1), 10, phase_train ) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second layer###
#
W_conv2 = weight_variable([5, 5, 10, 20])
b_conv2 = bias_variable([20])

h_conv2 = tf.nn.relu(batch_norm(conv2d(h_pool1, W_conv2), 20 , phase_train) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#####Third Layer #######

#W_conv3 = weight_variable([5,5,8,16])
#b_conv3 = bias_variable([16])
#h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#h_pool3 = max_pool_2x2(h_conv3)


#Fully Connected Layer#########
W_fc1 = weight_variable([14*16 *20, 5000])
b_fc1 = bias_variable([5000])

h_pool5_flat = tf.reshape(h_pool2, [-1, 14*16*20])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([5000, 2])
b_fc2 = bias_variable([2])

y_conv=tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Checkpoint variable
saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())
'''if os.path.exists("model.ckpt"):
    saver.restore(sess, "model.ckpt")
    print(sess.run(tf.all_variables()))
    print("Model restored.")
    print("test accuracy after reading from checkpoint file %g"%accuracy.eval(feed_dict={x: ds.test.images, y_: ds.test.labels, keep_prob: 1.0}))
else:
    print("train")
'''
for i in range(20):
	display_step = 0
        for batch_xs , batch_ys in ds.train.next_batch():
		display_step+=1
		if display_step%100 == 0:
                	train_accuracy = accuracy.eval(feed_dict={
                	x:batch_xs, y_: batch_ys, phase_train.name:False ,keep_prob: 1.0})
                	#print(sess.run(y_conv , feed_dict={x:batch[0]}))
                	print("step %d, training accuracy %g"%(display_step, train_accuracy))
		train_step.run(feed_dict={x: batch_xs, y_: batch_ys, phase_train.name : True , keep_prob: 0.7})
		
        	if display_step%200 == 0:
                	validation_accuracy = accuracy.eval(feed_dict={
                	x:ds.validation.images, y_: ds.validation.labels,phase_train.name:False,keep_prob: 1.0})
                	print("step %d, validation accuracy%g"%(display_step, validation_accuracy))
        	#train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
	save_path = saver.save(sess, "./model.ckpt")
	print("Model saved in file: %s" % save_path)
	print("test accuracy %g"%accuracy.eval(feed_dict={x: ds.test.images, y_: ds.test.labels, phase_train.name:False , keep_prob: 1.0}))
print("Done!!")
