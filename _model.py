from model4_cnn_lstm_0 import *

tanh    = tf.nn.tanh
sigmod  = tf.nn.sigmoid
relu    = tf.nn.relu
xavier_init = tf.contrib.layers.xavier_initializer
const_0_init = tf.constant_initializer
const_1_init = tf.constant_initializer

class Model:
  def __init__(
    self,
    max_seq_len: int,
    mfcc_num: int,
    rnn_layer_num: int,
    rnn_unit_num: int,
    attention_num: int,
    neg_sample_weight: float,
    is_training: bool
  ):
    self._is_training = is_training

    #(batch, height, width, channels)
    self.input_x = tf.placeholder(dtype=tf.float32,
                                  shape=[None, max_seq_len, mfcc_num, 3])
    self.input_y = tf.placeholder(dtype=tf.int32, shape=[None])
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")

    # input_x = tf.reshape(self.input_x, [-1, max_seq_len, mfcc_num * 3])

    layer1 = tf.layers.conv2d(
      inputs=self.input_x,
      filters=32,
      kernel_size=[6, 6],
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      strides=[1, 1],
      padding="VALID",
      activation=tf.nn.relu
    )
    layer1_max = tf.layers.max_pooling2d(
      inputs=layer1,
      pool_size=[3, 3],
      strides=[3, 3],
    )
    shape = layer1_max.shape.as_list()
    input_x = tf.reshape(layer1_max, [-1, shape[1], shape[2] * 32])

    outputs = TF.create_bi_LSTM(
      input_x, rnn_layer_num, rnn_unit_num, "lstm"
    )

    weighted_output = 0
    for atten_id in range(attention_num):
      with tf.name_scope(f"atten_id_{atten_id}"):
        context = tf.Variable(
          tf.random_uniform([2 * rnn_unit_num], -1., 1), dtype=tf.float32
        )
        w = tf.Variable(
          tf.random_uniform([], -1., 1.), dtype=tf.float32
        )
        weighted_status = TF.basic_attention(outputs, context)
        weighted_output += weighted_status * w

    if is_training:
      h_drop = tf.nn.dropout(weighted_output, self.dropout_keep_prob)
    else:
      h_drop = weighted_output

    logit = tf.layers.dense(h_drop, 2)
    self.class_probs = tf.nn.softmax(logit)
    self.predicted_class = tf.argmax(
      logit, 1, output_type=tf.int32
    )

    if is_training:
      y = tf.cast(self.input_y, tf.float32)
      weights = y + (1 - y) * neg_sample_weight
    else:
      weights = 1

    one_hot_y = tf.one_hot(self.input_y, 2)
    self.loss = tf.losses.softmax_cross_entropy(
      onehot_labels=one_hot_y,
      logits=logit,
      weights=weights,
      label_smoothing=0
    )

    self.error_count = tf.reduce_sum(
      tf.cast(
        tf.not_equal(self.input_y, tf.squeeze(self.predicted_class)), tf.int32
      )
    )

