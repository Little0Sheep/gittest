from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

x1=[]
y1=[]
index=668;
x1.append(mnist.test.images[index])
y1.append(mnist.test.labels[index])
# print(sess.run(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),feed_dict={x:batch_xs,y_:batch_ys}))
print("识别出图片中的数字为：",sess.run(tf.argmax(y,1)[0],feed_dict={x:x1}))
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print("准确率：%.2f%%" % (accuracy.eval({x:mnist.test.images,y_:mnist.test.labels})*100))

plt.imshow(np.array(x1[0]).reshape(28,28),plt.cm.gray)
plt.show()
# Image.fromarray(np.array(batch_xs[0]).reshape(28,28)).convert("L").save("2.jpg")
# plt.figure()
# plt.plot()
# plt.imshow(np.array(batch_xs[0]).reshape(28,28))
# plt.show()
# batch_xs,batch_ys=mnist.train.next_batch(2)
# print(batch_xs[0][783])
# print(batch_ys[0][9])