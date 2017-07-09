import os
import numpy as np
import struct
import PIL.Image
import scipy.misc
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt

train_data_dir="HWDB1.1trn_gnt"
test_data_dir="HWDB1.1tst_gnt"

def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size=10
        while True:
            header=np.fromfile(f,dtype="uint8",count=header_size)
            if not header.size: break
            sample_size=header[0]+(header[1]<<8)+(header[2]<<16)+(header[3]<<24)
            tagcode=header[5]+(header[4]<<8)
            width=header[6]+(header[7]<<8)
            height=header[8]+(header[9]<<8)
            if header_size+width*height !=sample_size:
                break
            image=np.fromfile(f,dtype="uint8",count=width*height).reshape((height,width))
            yield image,tagcode

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path=os.path.join(gnt_dir,file_name)
            with open(file_path,'rb') as f:
                for image,tagcode in one_file(f):
                    yield image,tagcode

# train_counter=0
# test_counter=0
# for image,tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
#     tagcode_unicode=struct.pack(">H",tagcode).decode("gb2312")
#
#     # if train_counter<1000:
#     #     im=PIL.Image.fromarray(image)
#     #     im.convert("RGB").save("pngs/"+tagcode_unicode+str(train_counter)+".png")
#
#     train_counter+=1
#
# for image,tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
#     tagcode_unicode=struct.pack(">H",tagcode).decode("gb2312")
#     test_counter+=1

# print(train_counter,test_counter)

char_set = "的一是了我不人在"
# 地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"

def resize_and_normalize_image(img):
    #补方
    pad_size=abs(img.shape[0]-img.shape[1])
    if img.shape[0]<img.shape[1]:
        pad_dims=((pad_size,pad_size),(0,0))
    else:
        pad_dims=((0,0),(pad_size,pad_size))
    img=np.lib.pad(img,pad_dims,mode="constant",constant_values=255)
    #缩放
    img=scipy.misc.imresize(img,(64-4*2,64-4*2))
    img=np.lib.pad(img,((4,4),(4,4)),mode="constant",constant_values=255)
    assert img.shape==(64,64)

    img=img.flatten()
    #像素范围 -1到1
    img=(img-128)/128
    return img

def convert_to_one_hot(char):
    vector=np.zeros(len(char_set))
    vector[char_set.index(char)]=1
    return vector

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={X:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={X:v_xs,Y:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    inital=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital=tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

def conv2d(x,W):
    # stride[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

train_data_x=[]
train_data_y=[]

counter=0
for image,tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode=struct.pack('>H',tagcode).decode("gb2312")
    if tagcode_unicode in char_set:
        train_data_x.append(resize_and_normalize_image(image))
        train_data_y.append(convert_to_one_hot(tagcode_unicode))
train_data_x,train_data_y=shuffle(train_data_x,train_data_y,random_state=0)

test_data_x=[]
test_data_y=[]
for image,tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode=struct.pack('>H',tagcode).decode("gb2312")
    if tagcode_unicode in char_set:
        test_data_x.append(resize_and_normalize_image(image))
        test_data_y.append(convert_to_one_hot(tagcode_unicode))
test_data_x,test_data_y=shuffle(test_data_x,test_data_y,random_state=0)

X=tf.placeholder(tf.float32,[None,64*64])
Y=tf.placeholder(tf.float32,[None,len(char_set)])
keep_prob=tf.placeholder(tf.float32)
x_image=tf.reshape(X,[-1,64,64,1])
#第一层卷积
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#第二层卷积
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#全连接第一层
W_fc1=weight_variable([16*16*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,16*16*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#全连接第二层
W_fc2=weight_variable([1024,len(char_set)])
b_fc2=bias_variable([len(char_set)])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(tf.clip_by_value(prediction,1e-10,1.0)),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver=tf.train.Saver()
# with tf.Session() as sess:
#     init=tf.global_variables_initializer()
#     sess.run(init)
#     for i in range(100):
#         sess.run(train_step,feed_dict={X:train_data_x,Y:train_data_y,keep_prob:0.5})
#         correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
#         accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#         print("准确率：%.2f%%" % (accuracy.eval({X:test_data_x,Y:test_data_y,keep_prob:1})*100))
#     save_path = saver.save(sess, "my_net/cn_cnn_net.ckpt")

with tf.Session() as sess:
    saver.restore(sess,"my_net/cn_cnn_net.ckpt")
    y_pre = sess.run(tf.argmax(prediction, 1), feed_dict={X: test_data_x[:5], keep_prob: 1})
    print("预测汉字是：",char_set[y_pre[0]])
    plt.imshow(np.array(test_data_x[:5]).reshape(64,64))
    plt.show()