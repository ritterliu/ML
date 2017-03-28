# -*- coding: utf-8 -*-
'''卷积神经网络测试MNIST数据'''
#导入MNIST数据
import input_data
import tensorflow as tf


from mnist_demo import * 
import os

mnist = input_data.read_data_sets2('Mnist_data/', one_hot=True)

model_filepath = "./model/cnn_model.ckpt"

sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# sess.run(tf.initialize_all_variables())
#权重初始化函数,用一个较小的正数来初始化偏置项
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#卷积和池化函数
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#把x变成一个4d向量
x_image = tf.reshape(x, [-1,28,28,1])
#把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
#进行池化。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#为了减少过拟合，在输出层之前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#添加一个softmax层，就像softmax regression一样
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#训练设置
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# sess.run(tf.initialize_all_variables())
#训练
# for i in range(100):
#   batch = mnist.train.next_batch(50)
#   if i%20 == 0:
#     train_accuracy = accuracy.eval(feed_dict={
#         x:batch[0], y_: batch[1], keep_prob: 1.0})
#     print "-->step %d, training accuracy %.4f"%(i, train_accuracy)
#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Save the model
# saver = tf.train.Saver()
# # save_path = saver.save(sess, model_filepath)

# saver.restore(sess, model_filepath)

saver = tf.train.Saver()
# save_path = saver.save(sess, model_filepath)

saver.restore(sess, model_filepath)

#最终评估
print "卷积神经网络测试MNIST数据集正确率: %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

dir_name="test_num"
files = os.listdir(dir_name)
cnt=len(files)

for i in range(cnt):
  files[i]=dir_name+"/"+files[i]
  print('files:', files[i])
  test_images1,test_labels1=GetImage([files[i]])

  input_index=int(files[i].strip().split('/')[1][0])
  print ("input_index:", input_index)

  # print (tf.cast(correct_prediction, tf.float32).eval)
  # print(shape(test_images1))
  mnist.test = input_data.DataSet(test_images1, test_labels1, dtype=tf.float32)

  print "------卷积神经网络测试MNIST数据集正确率: %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
  res=accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})

  # res = accuracy.eval(feed_dict={
  #   x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

  print('++++++++++++++++++++++++++++++++++++++++++')
  print(shape(mnist.test.images))
  print(shape(mnist.test.labels))
  print(shape(res))
  print('------------------------------------------')
  # print (tf.argmax(y, 1))
  # print(y.eval())
  # output_res = int(res[0])
  # print("output:",output_res)
  print("res:",res)
  print("\n")

'''

  if(output_res==input_index):
    correct = correct + 1
    print("correct!\n")
  else:
    wrong = wrong + 1
    print("wrong!\n")

  print('correct:',correct, ' wrong:',wrong)
  print(correct/(correct + wrong))
'''
  