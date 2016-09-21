import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Reshape the input
x_reshape = tf.reshape(x, [-1,28,28,1])

w_1 = weight_variable([5,5,1,32])
b_1 = bias_variable([32])

conv1 = conv2d(x_reshape, w_1)
relu1 = tf.nn.relu(conv1 + b_1)
pool1 = max_pool_2x2(relu1)

w_2 = weight_variable([5, 5, 32, 64])
b_2 = bias_variable([64])

conv2 = conv2d(pool1, w_2)
relu2 = tf.nn.relu(conv2 + b_2)
pool2 = max_pool_2x2(relu2)

#Fully connected stuff, reshapes from rows x 7*7*64 to rows x 1024
w_3 = weight_variable([7 * 7 * 64, 1024])
b_3 = bias_variable([1024])

flatten = tf.reshape(pool2, [-1, 7*7*64])
dense1 = tf.matmul(flatten, w_3)
relu3 = tf.nn.relu(dense1 + b_3)

#Connection to output layer
w_4 = weight_variable([1024, 10])
b_4 = bias_variable([10])

dense2 = tf.matmul(dense1, w_4)
y = tf.nn.softmax(dense2 + b_4)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Accuracy/output stuff
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(1000):
        batch = mnist.train.next_batch(100)
        out = sess.run([train_step], feed_dict={x: batch[0], y_: batch[1]})
        if i%100 == 0:
            print "Accuracy: {}".format(accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))

    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


