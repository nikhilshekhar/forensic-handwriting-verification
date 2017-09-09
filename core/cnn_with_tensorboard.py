import tensorflow as tf
import core.dataset

# Start the Session
sess = tf.InteractiveSession()

# Path for the dataset
ds = core.dataset.read_data_sets('/home/nikhil/data/scripts/', one_hot=True)

# Print dimensions of datasets
print("Train images dimensions:", ds.train.images.shape)
print("Train labels dimesnions:", ds.train.labels.shape)
print("Test images dimesnions:", ds.test.images.shape)
print("Test labels dimensions:", ds.test.labels.shape)

# Create placeholders
x = tf.placeholder(tf.float32, shape=[None, 4096])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
phase_train = tf.placeholder(tf.bool, name='phase_train')


# Initialise weight variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Initialize bias variables
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Define the convolution params
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# Define the max pooling params
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Define the batchnormalization method
def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


# ## First Layer ###
W_conv1 = weight_variable([3, 3, 1, 20])
b_conv1 = bias_variable([20])
x_image = tf.reshape(x, [-1, 64, 64, 1])
h_conv1 = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1), 20, phase_train) + b_conv1)

# ## Second layer ###
W_conv2 = weight_variable([3, 3, 20, 20])
b_conv2 = bias_variable([20])
h_conv2 = tf.nn.relu(batch_norm(conv2d(h_conv1, W_conv2), 20, phase_train) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)

# ## Third Layer ###
W_conv3 = weight_variable([3, 3, 20, 30])
b_conv3 = bias_variable([30])
h_conv3 = tf.nn.relu(batch_norm(conv2d(h_pool2, W_conv3), 30, phase_train) + b_conv3)

# ## Fourth Layer ###
W_conv4 = weight_variable([3, 3, 30, 30])
b_conv4 = bias_variable([30])
h_conv4 = tf.nn.relu(batch_norm(conv2d(h_conv3, W_conv4), 30, phase_train) + b_conv4)

# ## MAx pool 2 ###
h_pool3 = max_pool_2x2(h_conv4)

# ## First fully Connected Layer ###
W_fc1 = weight_variable([16 * 16 * 30, 3000])
b_fc1 = bias_variable([3000])

h_pool5_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ## Second fully connected Layer ###
W_fc2 = weight_variable([3000, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
tf.summary.scalar('cross_entropy', cross_entropy)

# Adam optimizer to minimize loss function
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

# Op to check if prediction was correct
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# Accuracy op
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Checkpoint variable
saver = tf.train.Saver()

# Tensorboard op to merge all events before writing to the event file
merged = tf.merge_all_summaries()

# Start the Session
sess = tf.InteractiveSession()

# Tensorboard writer
# summary_writer = tf.train.SummaryWriter('/home/nikhil/data/logs', graph=sess.graph)

# Initialize all variables
tf.initialize_all_variables().run()

# Tensorboard writer
summary_writer = tf.train.SummaryWriter('/home/nikhil/data/logs', graph=sess.graph)

display_step = 0
for i in range(25):
    print("******************* EPOCH %s **************" % i)
    for batch_xs, batch_ys in ds.train.next_batch():
        display_step += 1
        if display_step % 200 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={x: batch_xs, y_: batch_ys, phase_train.name: False, keep_prob: 1.0})
    validation_accuracy = accuracy.eval(
        feed_dict={x: ds.validation.images, y_: ds.validation.labels, phase_train.name: False, keep_prob: 1.0})
    print("step %d, training accuracy %g" % (display_step, train_accuracy))
print("step %d, validation accuracy%g" % (display_step, validation_accuracy))
# Actual training happens here
summary, _, loss = sess.run([merged, train_step, cross_entropy],
                            feed_dict={x: batch_xs, y_: batch_ys, phase_train.name: True, keep_prob: 0.7})
if display_step % 200 == 0:
    summary_writer.add_summary(summary, display_step)
# summary_writer.add_summary(summary, display_step)
# summary_writer.flush()
# summary,loss_val = sess.run([merged,cross_entropy] , feed_dict = {x: ds.train.images , y_: ds.train.labels ,
# phase_train.name : False , keep_prob : 1.0})
# summary_writer.add_summary(summary, i)
save_path = saver.save(sess, "./model.ckpt")
print("Model saved in file: %s" % save_path)
print("test accuracy %g" % accuracy.eval(
    feed_dict={x: ds.test.images, y_: ds.test.labels, phase_train.name: False, keep_prob: 1.0}))

print("Done!!")
# file_loss.close()
# accuracy_file.close()
summary_writer.flush()
summary_writer.close()
