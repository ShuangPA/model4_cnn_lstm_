from model4_cnn_lstm_P import *

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
    is_training: bool,
    T_embedding_size: int,
    T_filter_sizes: list,
    T_num_filters: int,
  ):
    self._is_training = is_training

    #(batch, height, width, channels)
    self.input_x = tf.placeholder(dtype=tf.float32,
                                  shape=[None, max_seq_len, mfcc_num, 3])
    self.input_y = tf.placeholder(dtype=tf.int32, shape=[None])
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")

    # input_x = tf.reshape(self.input_x, [-1, max_seq_len, mfcc_num * 3])

    ###
    self.input_phonemes = tf.placeholder(dtype=tf.int32, shape=[None, 200])
    with tf.name_scope("embedding"):
      self.W = tf.Variable(tf.random_uniform([130, T_embedding_size], -1, 1))
      self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_phonemes)
      self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    T_pooled_output = []
    for i, filter_size in enumerate(T_filter_sizes):
      with tf.name_scope(f"conv-maxpool-{filter_size}"):
        T_filter_shape = [filter_size, T_embedding_size, 1, T_num_filters]
        T_W = tf.Variable(tf.truncated_normal(T_filter_shape, stddev=0.1), name="T_W")
        T_b = tf.Variable(tf.constant(0.1, shape=[T_num_filters]), name="T_b")
        T_conv = tf.nn.conv2d(
          self.embedded_chars_expanded,
          T_W,
          strides=[1, 1, 1, 1],
          padding="VALID",
          name="T_conv"
        )
        T_h = tf.nn.relu(tf.nn.bias_add(T_conv, T_b), name="T_relu")
        T_pooled = tf.nn.max_pool(
          T_h,
          ksize=[1, 200 - filter_size + 1, 1, 1],
          strides=[1, 1, 1, 1],
          padding="VALID",
          name="T_pool"
        )
        T_pooled_output.append(T_pooled)
    num_filters_total = T_num_filters * len(T_filter_sizes)
    T_h_pool = tf.concat(T_pooled_output, 3)
    T_h_pool_flat = tf.reshape(T_h_pool, [-1, num_filters_total])

    ###

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

    ###
    weighted_output = tf.concat([weighted_output, T_h_pool_flat], 1)
    ###

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

