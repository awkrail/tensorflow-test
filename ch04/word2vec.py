import tensorflow as tf
import math
from data_set import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "data/", "Data set directory")
tf.app.flags.DEFINE_string("log_dir", "logs/", "Log directory")
tf.app.flags.DEFINE_integer("max_vocab", 2000, "Max Vocablary size.")
tf.app.flags.DEFINE_integer("skip_window", 2, "How many words to consider left and right")
tf.app.flags.DEFINE_integer("num_skips", 4, "How many times to reuse an input to generate a label")
tf.app.flags.DEFINE_integer("embedding_size", 64, "Dimension of the embedding sample")
tf.app.flags.DEFINE_integer("num_sumpled", 64, "Number of negative examples to sample")
tf.app.flags.DEFINE_integer("num_step", 10000, "Train step.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size.")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate")
tf.app.flags.DEFINE_bool("create_tsv", True, "Create words.tsv or not")

def main(argv):
  data = DataSet(FLAGS.data_dir, FLAGS.max_vocab)

  # 必要な値の取得
  batch_size = FLAGS.batch_size
  embedding_size = FLAGS.embedding_size
  vocab_size = len(data.w_to_id)

  # placeholderの定義
  inputs = tf.placeholder(tf.int32, shape=[batch_size])
  correct = tf.placeholder(tf.int32, shape=[batch_size, 1])

  # 単語の組み込みを取得
  word_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="word_embedding")
  embed = tf.nn.embedding_lookup(word_embedding, inputs)

  # NCE誤差(Negative-Sampling)
  w_out = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
  b_out = tf.Variable(tf.zeros([vocab_size]))

  nce_loss = tf.nn.nce_loss(weights=w_out, biases=b_out, labels=correct, inputs=embed, num_sampled=FLAGS.num_sampled, num_classes=vocab_size)
  loss = tf.reduce_mean(nce_loss)

  # 残り
  global_step = tf.Variable(0, name="global_step", trainable=False)
  train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver(max_to_keep=3)

  with tf.Session() as sess:
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_dir)
    if ckpt_state:
      last_model = ckpt_state.model_checkpoint_path
      saver.restore(sess, last_model)
      print("Model was loaded : ", last_model)
    else:
      sess.run(init)
      print("Initialized")
    
    last_step = sess.run(global_step)
    average_loss = 0
    
    for i in range(FLAGS.num_step):
      step = last_step + i + 1
      batch_inputs, batch_labels = data.create_next_batch(batch_size, FLAGS.num_skips, FLAGS.skip_window)
      feed_dict = {inputs: batch_inputs, correct: batch_labels}

      _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
      average_loss += loss_val

      if step % 100 == 0:
        average_loss /= 100
        print("Average Loss at step", step, ": ", average_loss)
        average_loss = 0
        saver.save(sess, FLAGS.log_dir + "my_model.ckpt", step)





if __name__ == "__main__":
  tf.app.run()