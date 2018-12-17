from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# mnistデータを格納したオブジェクトを呼び出す
mnist = input_data.read_data_sets("../ch02/data", one_hot=True)

"""モデル構築"""
# 入力データ
num_seq = 28
num_input = 28

x = tf.placeholder(tf.float32, [None, 784])
input = tf.reshape(x, [-1, num_seq, num_input])

# RNNモデルの構築
stacked_cells = []
for i in range(3):
  stacked_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=128))
cell = tf.nn.rnn.MultiRNNCell(cells=stacked_cells)

# dynamic_cellによる時間展開
output, _ = tf.nn.dynamic_rnn(cell=cell, inputs=input, dtype=tf.float32)

# RNNモデルの構築
last_output = output[:, -1, :]

w = tf.Variable(tf.truncated_normal([128, 10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))
out = tf.nn.softmax(tf.matmul(last_output, w) + b)

# 出力 / 誤差計算
y = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out + 1e-5), axis=[1]))

# 訓練
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 評価
correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# あとで家で実行する
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  test_images = mnist.test_images
  test_labels = mnist.test_labels

  for i in range(1000):
    train_images, train_labels = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x:train_images, y:train_labels})
    if i % 10 == 0:
      acc_val = sess.run(accuracy, feed_dict={x:test_images, y:test_labels})
      print("step ", i, " : ", acc_val)