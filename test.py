from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from PIL import Image
import numpy as np
import matp


flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('data_dir','./','directory for storing data')

mnist=input_data.read_data_sets(FLAGS.data_dir,one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_=tf.placeholder("float",[None,10])
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init=tf.initialize_all_variables()
sess=tf.InteractiveSession()
sess.run(init)
for i in range(100):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    # print(batch_ys[0])
    # print(sess.run(y,feed_dict={x:batch_xs,y_:batch_ys})[0][9])

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
print(mnist.test.images[0][783])
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
batch_xs,batch_ys=mnist.train.next_batch(1)
print(batch_ys[0])
print(batch_xs[0])
Image.fromarray(batch_xs[0])

# batch_xs,batch_ys=mnist.train.next_batch(2)
# print(batch_xs[0][783])
# print(batch_ys[0][9])