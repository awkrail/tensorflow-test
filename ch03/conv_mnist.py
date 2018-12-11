from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# mnistデータを呼び出す
mnist = input_data.read_data_sets("../ch02/data/", one_hot=True)

# 入力データを定義する
x = tf.placeholder(tf.float32, [None, 784])
# 入力画像をログに出力する
img = tf.reshape(x, [-1, 28, 28, 1])

# 畳み込み層1
f1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
conv1 = tf.nn.conv2d(img, f1, strides=[1, 1, 1, 1], padding='SAME')
b1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(conv1 + b1)
# プーリング層
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 畳み込み層2
f2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
conv2 = tf.nn.conv2d(h_pool1, f2, strides=[1, 1, 1, 1], padding='SAME')
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(conv2 + b2)
# プーリング層
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全結合層
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
w_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# 出力層
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
out = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

y = tf.placeholder(tf.float32, [None, 10])

# 誤差関数
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out + 1e-5), axis=[1])) # もう一度見直し

# 訓練
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 評価
correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

"""
実行
"""
with tf.Session() as sess:
  sess.run(init)
  # テストデータをロードする
  test_images = mnist.test.images
  test_labels = mnist.test.labels

  for i in range(1000):
    train_images, train_labels = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x:train_images, y:train_labels})

    if i % 100 == 0:
      acc_val = sess.run(accuracy, feed_dict={x:test_images, y:test_labels})
      print("step : ", i, " : accuracy : ", acc_val)