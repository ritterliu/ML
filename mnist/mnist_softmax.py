# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
# from tensorflow.examples.tutorials.mnist import input_data

import input_data

import tensorflow as tf

import os

from mnist_demo import * 

flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('data_dir', 'data/', 'Directory for storing data')
flags.DEFINE_string('data_dir', 'Mnist_data/', 'Directory for storing data')

mnist = input_data.read_data_sets2(FLAGS.data_dir, one_hot=True)

#try
# sess = tf.InteractiveSession()


# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
## try
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#try
sess = tf.InteractiveSession()

# Train

#try
# tf.initialize_all_variables().run()
tf.global_variables_initializer().run()

print('----Train-----')
# print(tf.argmax(W,1).eval())
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # print(tf.argmax(W,1).eval())
  #try
  # train_step.run({x: batch_xs, y_: batch_ys})
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # break

# Test trained model
print('Test trained model----------------')
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

#try
# accuracy2 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print('==================================accuracy2:', sess.run(accuracy2, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy=tf.cast(tf.argmax(y, 1),tf.float32)
# accuracy=y
# print((mnist.test.labels))
print('---accuracy:', accuracy);


dir_name="test_num"
files = os.listdir(dir_name)
cnt=len(files)


# Test trained model
# test_images1,test_labels1=GetImage2(files, dir_name)
# mnist.test = input_data.DataSet(test_images1, test_labels1, dtype=tf.float32, reshape=reshape)
accuracy3 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('===============================accuracyA:', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print('===============================accuracyB:', sess.run(accuracy3, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

correct = 0
wrong = 0

for i in range(cnt):
  files[i]=dir_name+"/"+files[i]
  print('files:', files[i])
  test_images1,test_labels1=GetImage([files[i]])

  input_index=int(files[i].strip().split('/')[1][0])
  print ("input_index:", input_index)

  # print (tf.cast(correct_prediction, tf.float32).eval)
  # print(shape(test_images1))
  mnist.test = input_data.DataSet(test_images1, test_labels1, dtype=tf.float32)
  res=accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})

  # print('++++++++++++++++++++++++++++++++++++++++++')
  # print(shape(mnist.test.images))
  # print('------------------------------------------')
  # print (tf.argmax(y, 1))
  # print(y.eval())
  output_res = int(res[0])
  print("output:",output_res)
  print("\n")

  # if(output_res==input_index):
  #   correct = correct + 1
  #   print("correct!\n")
  # else:
  #   wrong = wrong + 1
  #   print("wrong!\n")

  # print('correct:',correct, ' wrong:',wrong)
  # print(correct/(correct + wrong))

  # print("\n")



